import copy
import csv

import numpy as np
import numpy.linalg as LA
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

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
        self.aug = RGBDAugmentor(reshape_size=reshape_size)

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

    def normalize_safe_np(self, v, axis=-1, eps=1e-6):
        de = LA.norm(v, axis=axis, keepdims=True)
        de = np.maximum(de, eps)
        return v / de

    def segs2lines_np(self, segs):
        ones = np.ones(len(segs))
        ones = np.expand_dims(ones, axis=-1)
        p1 = np.concatenate([segs[:, :2], ones], axis=-1)
        p2 = np.concatenate([segs[:, 2:], ones], axis=-1)
        lines = np.cross(p1, p2)
        return self.normalize_safe_np(lines)

    def normalize_segs(self, lines, pp, rho=517.97):
        pp = np.array([pp[0], pp[1], pp[0], pp[1]], dtype=np.float32)
        return (lines - pp)/rho

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
        return sampled_segs

    def coordinate_yup(self, segs, org_h):
        H = np.array([0, org_h, 0, org_h])
        segs[:, 1] = -segs[:, 1]
        segs[:, 3] = -segs[:, 3]
        return (H + segs)

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
        lines[0] = self.sample_segs_np(lines[0], num_sample=512)
        endpoint.append(lines[0])
        lines[0] = self.segs2lines_np(lines[0])

        lines[1] = self.coordinate_yup(lines[1], sizey)
        lines[1] = self.normalize_segs(lines[1], pp=pp, rho=rho)
        lines[1] = self.sample_segs_np(lines[1], num_sample=512)
        endpoint.append(lines[1])
        lines[1] = self.segs2lines_np(lines[1])

        images = F.interpolate(images, size=(sizey, sizex), mode="bilinear")
        lines = np.array(lines)
        vps = np.array(vps)
        endpoint = np.array(endpoint)

        return images, poses, intrinsics, lines, vps, endpoint

    def __getitem__(self, index):
        target = {}
        images_list = self.scene_info["images"][index]
        poses = self.scene_info["poses"][index]
        intrinsics = self.scene_info["intrinsics"][index]
        lines_list = self.scene_info["lines"][index]
        vp_list = self.scene_info['vps'][index]

        images = []
        for i in range(2):
            images.append(self.image_read(images_list[i]))
            
        org_img0 = images[0]
        org_img1 = images[1]

        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)

        images = np.stack(images).astype(np.float32)
        images = torch.from_numpy(images).float() # [2,480,640,3] => [img_num,h,w,c]
        images = images.permute(0, 3, 1, 2)  # [2,3,480,640] => [img_num,c,h,w]

        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)
        lines = copy.deepcopy(lines_list)

        vps = []
        for i in range(2):
            vps.append(np.array(vp_list[i]))
        images = self.aug(
            images
        )
        images, poses, intrinsics, lines, vps, endpoint = self.process_geometry(
            images, poses, intrinsics, lines, vps)
        
        
        target['vps'] = (
            torch.from_numpy(np.ascontiguousarray(vps)).contiguous().float()
        )
        target['poses'] = (
            torch.from_numpy(np.ascontiguousarray(poses)).contiguous().float()
        )
        target['endpoint'] = (
            torch.from_numpy(np.ascontiguousarray(endpoint)).contiguous().float()
        )
        target['intrinsics'] = (
            torch.from_numpy(np.ascontiguousarray(intrinsics)).contiguous().float()
        )
        
        
        target['org_img0'] = org_img0
        target['org_img1'] = org_img1
        target['img_path0'] = images_list[0]
        target['img_path1'] = images_list[1]
        
        return images, lines, target
        

    def __len__(self):
        return len(self.scene_info["images"])
