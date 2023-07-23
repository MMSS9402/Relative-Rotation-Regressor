import csv
import numpy as np
import numpy.linalg as LA
import cv2
import torch
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

        self.aug = RGBDAugmentor(reshape_size=reshape_size)

        self.matterport = False
        if "matterport" in data_path:
            self.matterport = True
            self.scene_info = self._build_dataset()
        elif "StreetLearn" in self.name or "InteriorNet" in self.name:
            self.use_mini_dataset = use_mini_dataset
            self.scene_info = self._build_dataset()
        else:
            print("not currently setup in case have other dataset type!")
            import pdb
            pdb.set_trace()

    def _build_dataset(self):
        raise NotImplementedError

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    def read_line_file(self,filename, min_line_length=10):
        segs = []  # line segments

        with open(str(filename), "r") as csvfile:
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

    def __getitem__(self, index):
        """return training video"""
        if self.matterport:
            images_list = self.scene_info["images"][index]
            poses = self.scene_info["poses"][index]
            intrinsics = self.scene_info["intrinsics"][index]
            lines_list = self.scene_info["lines"][index]
            vp_list = self.scene_info['vps'][index]

            images = []
            for i in range(2):
                images.append(self.__class__.image_read(images_list[i]))

            poses = np.stack(poses).astype(np.float32)
            intrinsics = np.stack(intrinsics).astype(np.float32)

            images = np.stack(images).astype(np.float32)
            images = torch.from_numpy(
                images
            ).float()  # [2,480,640,3] => [img_num,w,h,c]
            images = images.permute(0, 3, 1, 2)  # [2,3,480,640] => [img_num,c,w,h]

            poses = torch.from_numpy(poses)
            intrinsics = torch.from_numpy(intrinsics)
            lines = []
            for i in range(2):
                lines.append(self.read_line_file(lines_list[i],10)) 

            vps = []
            for i in range(2):
                vps.append(np.array(vp_list[i]))
            images, poses, intrinsics, lines, vps = self.aug(
                images, poses, intrinsics, lines, vps
            )

            return images, poses, intrinsics, lines, vps
        else:
            local_index = index
            # in case index fails
            while True:
                try:
                    images_list = self.scene_info["images"][local_index]
                    poses = self.scene_info["poses"][local_index]
                    intrinsics = self.scene_info["intrinsics"][local_index]
                    lines_list = self.scene_info["lines"][local_index]
                    vp_list = self.scene_info['vps'][index]
                    images = []
                    
                    for i in range(2):
                        images.append(self.__class__.image_read(images_list[i]))
                    
                    poses = np.stack(poses).astype(np.float32)
                    intrinsics = np.stack(intrinsics).astype(np.float32)

                    images = np.stack(images).astype(np.float32)
                    images = torch.from_numpy(images).float()
                    images = images.permute(0, 3, 1, 2)

                    poses = torch.from_numpy(poses)
                    intrinsics = torch.from_numpy(intrinsics)
                    lines = []
                    for i in range(2):
                        lines.append(self.__class__.read_line_file(lines_list[i], 10))

                    vps = []
                    for i in range(2):
                        vps.append(np.array(vp_list[i]))

                    images, poses, intrinsics, lines, vps = self.aug(
                        images, poses, intrinsics, lines,vps
                    )

                    return images, poses, intrinsics, lines,vps
                except:
                    local_index += 1
                    continue

    def __len__(self):
        return len(self.scene_info["images"])
