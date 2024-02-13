import copy
import csv
import h5py
import numpy as np
import numpy.linalg as LA
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .augmentation import RGBDAugmentor


class RGBDDataset(Dataset):
    def __init__(
        self,
        name: str,
        data_path: str,
        ann_filename: str,
        reshape_size: (int, int) = (480, 640),
        use_mini_dataset: bool = False,
    ):
        """Base class for RGBD dataset"""
        self.name = name
        self.data_path = data_path
        self.ann_filename = ann_filename

        self.output_size = reshape_size
        self.augcolor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ColorJitter(
                    brightness=0.25, contrast=0.25, saturation=0.25, hue=0.4 / 3.14
                ),
                transforms.RandomGrayscale(p=0.1),
            ]
        )

        self.matterport = False
        if "Matterport" in name:
            self.matterport = True
            self.scene_info = self._build_dataset()
        elif "StreetLearn" in self.name or "InteriorNet" in self.name:
            self.use_mini_dataset = use_mini_dataset
            self.scene_info = self._build_dataset()
        else:
            raise f"not currently setup in case have other dataset type {name}!"

    def _build_dataset(self):
        raise NotImplementedError
    
    def load_h5py_to_dict(self, file_path):
        with h5py.File(file_path, 'r') as f:
            return {key: torch.tensor(f[key][:]) for key in f.keys()}

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    def read_line_file(self, filename: str, min_line_length=10):
        segs = []  # line segments

        with open(filename, "r") as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                segs.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        segs = np.array(segs, dtype=np.float32)
        lengths = LA.norm(segs[:, 2:] - segs[:, :2], axis=1)
        segs = segs[lengths > min_line_length]
        return segs

    def sample_segs_np(self, segs, num_sample):
        num_segs = len(segs)
        sampled_segs = np.zeros([num_sample, 4], dtype=np.float32)
        mask = np.zeros([num_sample, 1], dtype=np.float32)
        if num_sample > num_segs:
            sampled_segs[:num_segs] = segs
            mask[:num_segs] = np.ones([num_segs, 1], dtype=np.float32)
        else:
            lengths = LA.norm(segs[:, 2:] - segs[:, :2], axis=-1)
            prob = lengths / np.sum(lengths)
            idxs = np.random.choice(segs.shape[0], num_sample, replace=True, p=prob)
            sampled_segs = segs[idxs]
            mask = np.ones([num_sample, 1], dtype=np.float32)
        return sampled_segs, mask

    def coordinate_yup(self, segs, org_h):
        H = np.array([0, org_h, 0, org_h])
        segs[:, 1] = -segs[:, 1]
        segs[:, 3] = -segs[:, 3]
        return (H + segs)
    
    def normalize_segs(self,lines, pp, rho=517.97):
        pp = np.array([pp[0], pp[1], pp[0], pp[1]], dtype=np.float32)
        return (lines - pp)/rho
    
    def normalize_safe_np(self,v, axis=-1, eps=1e-6):
        de = LA.norm(v, axis=axis, keepdims=True)
        de = np.maximum(de, eps)
        return v / de
    
    def segs2lines_np(self,segs):
        ones = np.ones(len(segs))
        ones = np.expand_dims(ones, axis=-1)
        p1 = np.concatenate([segs[:, :2], -ones], axis=-1)
        p2 = np.concatenate([segs[:, 2:], -ones], axis=-1)
        lines = np.cross(p1, p2)
        return self.normalize_safe_np(lines)

    def process_geometry(self, images, poses, intrinsics, lines, vps):
        endpoint = []

        sizey, sizex = self.output_size  # (480, 640)
        scalex = sizex / images.shape[-1]
        scaley = sizey / images.shape[-2]

        xidx = np.array([0, 2])
        yidx = np.array([1, 3])
        intrinsics[:, xidx] = scalex * intrinsics[:, xidx]
        intrinsics[:, yidx] = scaley * intrinsics[:, yidx]

        pp = (images.shape[-1] / 2, images.shape[-2] / 2)  # 320, 240
        # rho = 2.0 / np.minimum(images.shape[-2], images.shape[-1])
        rho = 517.97  # focal length of matterport dataset

        lines[0] = self.coordinate_yup(lines[0], sizey)
        lines[0] = self.normalize_segs(lines[0], pp=pp, rho=rho)
        lines[0], line0_mask = self.sample_segs_np(lines[0], num_sample=512)
        endpoint.append(lines[0])
        lines[0] = self.segs2lines_np(lines[0])

        lines[1] = self.coordinate_yup(lines[1], sizey)
        lines[1] = self.normalize_segs(lines[1], pp=pp, rho=rho)
        lines[1], line1_mask = self.sample_segs_np(lines[1], num_sample=512)
        endpoint.append(lines[1])
        lines[1] = self.segs2lines_np(lines[1])
        
        line_mask = np.stack([line0_mask, line1_mask], axis=0) #[num,line_num,1]
        line_mask = line_mask.astype(np.float32)

        images = F.interpolate(images, size=(sizey, sizex), mode="bilinear")
        lines = np.array(lines).astype(np.float32)
        vps = np.array(vps)
        endpoint = np.array(endpoint)

        return images, poses, intrinsics, lines, vps, endpoint ,line_mask

    def __getitem__(self, index):
        target = {}
        images_list = self.scene_info["images"][index]
        poses = self.scene_info["poses"][index]
        intrinsics = self.scene_info["intrinsics"][index]
        # lines_list = self.scene_info["lines"][index]
        vp_list = self.scene_info['vps'][index]

        images = []
        h5py_path = []
        
        for i in range(2):
            images.append(self.image_read(images_list[i]))
            h5py_path.append(self.load_h5py_to_dict(images_list[i].replace(".png", "_sp_line.h5py",)))
        
        org_img0 = images[0].copy()
        org_img1 = images[1].copy()
        
        angle_sublines0 = h5py_path[0]['angle_sublines'][0].float()
        angles0 = h5py_path[0]['angles'][0].float()
        desc_sublines0 = h5py_path[0]['desc_sublines'][0].float()
        klines0 = h5py_path[0]['klines'][0].float()
        length_klines0 = h5py_path[0]['length_klines'][0].float()
        mask_sublines0 = h5py_path[0]['mask_sublines'][0].float()
        mat_klines2sublines0 = h5py_path[0]['mat_klines2sublines'][0].float()
        num_klns0 = h5py_path[0]['num_klns'][0].float()
        num_slns0 = h5py_path[0]['num_slns'][0].float()
        pnt_sublines0 = h5py_path[0]['pnt_sublines'][0].float()
        resp_sublines0 = h5py_path[0]['resp_sublines'][0].float()
        score_sublines0 = h5py_path[0]['score_sublines'][0].float()
        sublines0 = h5py_path[0]['sublines'][0].float()
        num_segs = sublines0.shape[0]
        
        angle_sublines1 = h5py_path[1]['angle_sublines'][0].float()
        angles1 = h5py_path[1]['angles'][0].float()
        desc_sublines1 = h5py_path[1]['desc_sublines'][0].float()
        klines1 = h5py_path[1]['klines'][0].float()
        length_klines1 = h5py_path[1]['length_klines'][0].float()
        mask_sublines1 = h5py_path[1]['mask_sublines'][0].float()
        mat_klines2sublines1 = h5py_path[1]['mat_klines2sublines'][0].float()
        num_klns1 = h5py_path[1]['num_klns'][0].float()
        num_slns1 = h5py_path[1]['num_slns'][0].float()
        pnt_sublines1 = h5py_path[1]['pnt_sublines'][0].float()
        resp_sublines1 = h5py_path[1]['resp_sublines'][0].float()
        score_sublines1 = h5py_path[1]['score_sublines'][0].float()
        sublines1 = h5py_path[1]['sublines'][0].float()

        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)
        
        # images = torch.from_numpy(images).float() # [2,480,640,3] => [img_num,h,w,c]
        # images = images.permute(0, 3, 1, 2)  # [2,3,480,640] => [img_num,c,h,w]

        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)
        # lines = copy.deepcopy(lines_list)

        vps = []
        for i in range(2):
            vps.append(np.array(vp_list[i]))
        
        images[0] = images[0][:,:,::-1].astype(np.float32)
        images[1] = images[1][:,:,::-1].astype(np.float32)
        

        images[0] = self.augcolor(images[0] / 255.0)
        images[1] = self.augcolor(images[1] / 255.0)
        
        images = torch.stack(images)

        height, width = images[0].shape[-2],images[0].shape[-1]
        pp = (width / 2, height / 2)
        rho = 517.97

        lines0 = np.copy(klines0.reshape(num_segs,-1).numpy())

        lines0 = self.coordinate_yup(lines0,height)
        lines0 = self.normalize_segs(lines0, pp=pp, rho=rho)
        lines0,line_mask0 = self.sample_segs_np(lines0, num_sample=512)
        normal0 = self.segs2lines_np(lines0)

        lines1 = np.copy(klines1.reshape(num_segs,-1).numpy())
        lines1 = self.coordinate_yup(lines1,height)
        lines1 = self.normalize_segs(lines1, pp=pp, rho=rho)
        lines1,line_mask1 = self.sample_segs_np(lines1, num_sample=512)
        normal1 = self.segs2lines_np(lines1)
        
        target['vps'] = (
            torch.from_numpy(np.ascontiguousarray(vps)).contiguous().float()
        )
        target['poses'] = (
            torch.from_numpy(np.ascontiguousarray(poses)).contiguous().float()
        )
        target['endpoint'] = (
            torch.from_numpy(np.ascontiguousarray(lines1)).contiguous().float()
        )
        target['intrinsics'] = (
            torch.from_numpy(np.ascontiguousarray(intrinsics)).contiguous().float()
        )
        target['normal0'] = (
            torch.from_numpy(np.ascontiguousarray(normal0)).contiguous().float()
        )  
        target['normal1'] = (
            torch.from_numpy(np.ascontiguousarray(normal1)).contiguous().float()
        )
        target['lines0'] = (
            torch.from_numpy(np.ascontiguousarray(lines0)).contiguous().float()
        )
        target['lines1'] = (
            torch.from_numpy(np.ascontiguousarray(lines1)).contiguous().float()
        )

        target['angle_sublines0'] = angle_sublines0
        target['angles0'] = angles0
        target['desc_sublines0'] = desc_sublines0
        target['klines0'] = klines0
        target['length_klines0'] = length_klines0
        target['mask_sublines0'] = mask_sublines0
        target['mat_klines2sublines0'] = mat_klines2sublines0
        target['num_klns0'] = num_klns0
        target['num_slns0'] = num_slns0
        target['pnt_sublines0'] = pnt_sublines0
        target['resp_sublines0'] = resp_sublines0
        target['score_sublines0'] = score_sublines0
        target['sublines0'] = sublines0
        
        target['angle_sublines1'] = angle_sublines1
        target['angles1'] = angles1
        target['desc_sublines1'] = desc_sublines1
        target['klines1'] = klines1
        target['length_klines1'] = length_klines1
        target['mask_sublines1'] = mask_sublines1
        target['mat_klines2sublines1'] = mat_klines2sublines1
        target['num_klns1'] = num_klns1
        target['num_slns1'] = num_slns1
        target['pnt_sublines1'] = pnt_sublines1
        target['resp_sublines1'] = resp_sublines1
        target['score_sublines1'] = score_sublines1
        target['sublines1'] = sublines1
        
        target['org_img0'] = org_img0
        target['org_img1'] = org_img1
        target['img_path0'] = images_list[0]
        target['img_path1'] = images_list[1]
        target['h5py_path0'] = h5py_path[0]
        target['h5py_path1'] = h5py_path[1]

        target['lmask0'] = line_mask0
        target['lmask1'] = line_mask1
        
        return images , target
        

    def __len__(self):
        return len(self.scene_info["images"])