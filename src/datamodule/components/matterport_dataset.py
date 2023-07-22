import os
import csv
import pickle

import numpy as np
import hydra
from lightning import LightningDataModule
import torch
import cv2
import numpy.linalg as LA
from torch.utils.data import Dataset

import json


class MatterportDataset(Dataset):

    # scale depth to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(
            self,
            root_path: str,
            ann_filename: str,
    ):
        super(MatterportDataset, self).__init__()

        base_pose = np.array([0, 0, 0, 0, 0, 0, 1])

        with open(os.path.join(root_path, "mp3d_planercnn_json", ann_filename)) as file:
            split = json.load(file)

        images = []
        lines = []
        vps = []
        poses = []
        intrinsics = []

        for data in split["data"]:

            for img_idx in ["0", "1"]:
                img_name = os.path.join(root_path,
                                        "/".join(data[img_idx]["file_name"].split("/")[6:]))

                line_name = img_name.split("/")
                line_name[9] = img_name.split("/")[9].split(".")[0] + "_line.csv"
                line_name = "/".join(line_name)

                vp1 = data[img_idx]['vp1']
                vp2 = data[img_idx]['vp2']
                vp3 = data[img_idx]['vp3']

                gt_vps = np.array([vp1, vp2, vp3])
                vps.append(gt_vps)

                images.append(img_name)
                lines.append(line_name)

            rel_pose = np.array(data["rel_pose"]["position"] + data["rel_pose"]["rotation"])

            # on matterport, we scale depths to balance rot & trans loss
            rel_pose[:3] /= self.DEPTH_SCALE

            # swap 3 & 6, we want W last for consistency with our other datasets
            rel_pose[6], rel_pose[3] = rel_pose[3], rel_pose[6]

            # normalize quaternions to have positive "W"
            if rel_pose[6] < 0:
                rel_pose[3:] *= -1
            poses = np.vstack([base_pose, rel_pose])

            intrinsics = np.array(
                [[517.97, 517.97, 320, 240], [517.97, 517.97, 320, 240]]
            )  # 480 x 640 imgs

        self.scene_info = {
            "images": images,
            "poses": poses,
            "intrinsics": intrinsics,
            "lines": lines,
            'vps': vps,
        }

    def __len__(self):
        return len(self.list_img_filename)

    def __getitem__(self, index):
        pass

