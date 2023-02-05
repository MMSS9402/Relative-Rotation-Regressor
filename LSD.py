import cv2
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp
import pickle
import json
import pandas as pd
import tqdm

scene_info = {"images": [], "poses": [], "intrinsics": []}
base_pose = np.array([0, 0, 0, 0, 0, 0, 1])
path = "matterport/mp3d_planercnn_json/cached_set_train.json"
DEPTH_SCALE = 5.0
root = "C:/Users/User/source/CuTi/matterport/rgb/"

with open(osp.join(path)) as f:
    split = json.load(f)

for i in range(len(split["data"])):
    images = []
    for imgnum in ["0", "1"]:
        img_name = os.path.join(
            root,
            "/".join(str(split["data"][i][imgnum]["file_name"]).split("/")[6:]),
        )

        img = cv2.imread(img_name, 0)
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(img)[0]

        drawn_img = lsd.drawSegments(img, lines)

        Line_image_name = img_name.split("/")
        csv_file_name = img_name.split("/")
        Line_image_name[9] = (
            img_name.split("/")[9].split(".")[0]
            + "_line."
            + img_name.split("/")[9].split(".")[1]
        )
        csv_file_name[9] = img_name.split("/")[9].split(".")[0] + "_line.csv"
        Line_image_name = "/".join(Line_image_name)
        csv_file_name = "/".join(csv_file_name)
        print(Line_image_name)
        cv2.imwrite(Line_image_name, drawn_img)
        data_df = pd.DataFrame(lines[0][0])
        data_df = data_df.T
        for i in range(1, len(lines)):
            data_df2 = pd.DataFrame(lines[i][0])
            data_df2 = data_df2.T
            data_df = pd.concat([data_df, data_df2])
        print(csv_file_name)
        data_df.to_csv(csv_file_name, index=False, header=None)
