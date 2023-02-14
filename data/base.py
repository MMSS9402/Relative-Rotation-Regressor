import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy.linalg as LA
import csv
import os
import cv2
import math
import random
import json
import pickle
import os.path as osp
import time

from .augmentation import RGBDAugmentor


class RGBDDataset(data.Dataset):
    def __init__(
        self,
        name,
        datapath,
        reshape_size=[384, 512],
        subepoch=None,
        is_training=True,
        gpu=0,
        streetlearn_interiornet_type=None,
        use_mini_dataset=False,
    ):
        """Base class for RGBD dataset"""
        self.root = datapath
        self.name = name
        self.streetlearn_interiornet_type = streetlearn_interiornet_type

        self.aug = RGBDAugmentor(reshape_size=reshape_size, datapath=datapath)

        self.matterport = False
        if "matterport" in datapath:
            self.matterport = True
            self.scene_info = self._build_dataset(subepoch == 10)
        elif "StreetLearn" in self.name or "InteriorNet" in self.name:
            self.use_mini_dataset = use_mini_dataset
            self.scene_info = self._build_dataset(subepoch)
        else:
            print("not currently setup in case have other dataset type!")
            import pdb

            pdb.set_trace()

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    def read_line_file(self,filename, min_line_length=10):
        segs = []  # line segments
        # csv 파일 열어서 Line 정보 가져오기
        print("filename:",filename)
        with open(str(filename), "r") as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                segs.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        segs = np.array(segs, dtype=np.float32)
        lengths = LA.norm(segs[:, 2:] - segs[:, :2], axis=1)
        segs = segs[lengths > min_line_length]
        return segs


    def normalize_safe_np(self,v, axis=-1, eps=1e-6):
        de = LA.norm(v, axis=axis, keepdims=True)
        de = np.maximum(de, eps)
        return v / de


    def segs2lines_np(self,segs):
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
            images, poses, intrinsics, lines = self.aug(
                images, poses, intrinsics, lines
            )

            return images, poses, intrinsics, lines
        else:
            local_index = index
            # in case index fails
            while True:
                try:
                    images_list = self.scene_info["images"][local_index]
                    poses = self.scene_info["poses"][local_index]
                    intrinsics = self.scene_info["intrinsics"][local_index]
                    lines_list = self.scene_info["lines"][local_index]

                    images = []
                    
                    for i in range(2):
                        images.append(self.__class__.image_read(images_list[i]))
                    print("out2:", images)
                    poses = np.stack(poses).astype(np.float32)
                    intrinsics = np.stack(intrinsics).astype(np.float32)

                    images = np.stack(images).astype(np.float32)
                    images = torch.from_numpy(images).float()
                    images = images.permute(0, 3, 1, 2)

                    poses = torch.from_numpy(poses)
                    intrinsics = torch.from_numpy(intrinsics)
                    lines = []
                    print("lines_list:",lines_list)
                    for i in range(2):
                        lines.append(self.__class__.read_line_file(lines_list[i], 10))

                    images, poses, intrinsics, lines = self.aug(
                        images, poses, intrinsics, lines
                    )

                    return images, poses, intrinsics, lines
                except:
                    local_index += 1
                    continue

    def __len__(self):
        return len(self.scene_info["images"])
