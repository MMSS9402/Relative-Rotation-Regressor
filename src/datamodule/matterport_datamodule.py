import os
import glob


import numpy as np
import hydra
from lightning import LightningDataModule
import torch
import cv2
import pickle
import csv
import numpy.linalg as LA


import json


class MatterportDatamodule(LightningDataModule):
    def __init__(self, ):
        super(MatterportDatamodule, self).__init__()

    def setup(self, stage: str) -> None:
        pass

    def _build_dataset(self, valid=False):


        print("Building Matterport dataset")

        scene_info = {
            "images": [],
            "poses": [],
            "intrinsics": [],
            "lines": [],
            'vps': [],
        }  # line 정보 추가
        base_pose = np.array([0, 0, 0, 0, 0, 0, 1])

        path = "cached_set_moon_train_vp.json"
        # if valid:
        #     print("valid data load!!")
        #     path = "cached_set_val.json"
        # if self.mode == "test":
        #     print("load_cached_test_data_json!!")
        #     path = "cached_set_test.json"
        with open(osp.join(self.root, "mp3d_planercnn_json", path)) as f:
            split = json.load(f)

        for i in range(len(split["data"])):
            images = []
            lines = []
            vp1 = []
            vp2 = []
            vps = []
            for imgnum in ["0", "1"]:
                img_name = os.path.join(
                    self.root,
                    "/".join(str(split["data"][i][imgnum]["file_name"]).split("/")[6:]),
                )
                line_name = img_name.split("/")
                line_name[9] = img_name.split("/")[9].split(".")[0] + "_line.csv"
                line_name = "/".join(line_name)
                vp1 = split["data"][i][imgnum]['vp1']
                vp2 = split["data"][i][imgnum]['vp2']
                vp3 = split["data"][i][imgnum]['vp3']
                gt_vps = np.array([vp1,vp2,vp3])
                vps.append(gt_vps)
                images.append(img_name)
                lines.append(line_name)

            rel_pose = np.array(
                split["data"][i]["rel_pose"]["position"]
                + split["data"][i]["rel_pose"]["rotation"]
            )
            og_rel_pose = np.copy(rel_pose)

            # on matterport, we scale depths to balance rot & trans loss
            rel_pose[:3] /= Matterport.DEPTH_SCALE
            cprp = np.copy(rel_pose)
            rel_pose[6] = cprp[
                3
            ]  # swap 3 & 6, we want W last for consistency with our other datasets
            rel_pose[3] = cprp[6]
            if rel_pose[6] < 0:  # normalize quaternions to have positive "W"
                rel_pose[3:] *= -1
            poses = np.vstack([base_pose, rel_pose])

            intrinsics = np.array(
                [[517.97, 517.97, 320, 240], [517.97, 517.97, 320, 240]]
            )  # 480 x 640 imgs

            scene_info["images"].append(images)
            scene_info["poses"] += [poses]
            scene_info["intrinsics"] += [intrinsics]
            scene_info["lines"].append(lines)
            scene_info['vps'].append(vps)

        return scene_info

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)
