import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp
import pickle
import csv
import numpy.linalg as LA

from .base_test import RGBDDataset
import json


class Matterport(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode="training", **kwargs):
        self.mode = mode

        super(Matterport, self).__init__(name="Matterport", **kwargs)

    def _build_dataset(self,valid=False):
        np.seterr(all="ignore")
        from tqdm import tqdm

        print("Building Matterport dataset")

        scene_info = {
            "images": [],
            "poses": [],
            "intrinsics": [],
            "lines": [],
        }  # line 정보 추가
        base_pose = np.array([0, 0, 0, 0, 0, 0, 1])

        path = "cached_set_test.json"
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
            for imgnum in ["0", "1"]:
                img_name = os.path.join(
                    self.root,
                    "/".join(str(split["data"][i][imgnum]["file_name"]).split("/")[6:]),
                )
                line_name = img_name.split("/")
                line_name[8] = img_name.split("/")[8].split(".")[0] + "_line.csv"
                line_name = "/".join(line_name)

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

        return scene_info

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)
