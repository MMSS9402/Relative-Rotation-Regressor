
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp
import pickle
import json
import pandas as pd
from tqdm import tqdm

import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp
import pickle
import csv
import numpy.linalg as LA
import data.transforms as T


def eul2rotm_ypr(euler):
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(euler[0]), -np.sin(euler[0])],
            [0, np.sin(euler[0]), np.cos(euler[0])],
        ],
        dtype=np.float32,
    )

    R_y = np.array(
        [
            [np.cos(euler[1]), 0, np.sin(euler[1])],
            [0, 1, 0],
            [-np.sin(euler[1]), 0, np.cos(euler[1])],
        ],
        dtype=np.float32,
    )

    R_z = np.array(
        [
            [np.cos(euler[2]), -np.sin(euler[2]), 0],
            [np.sin(euler[2]), np.cos(euler[2]), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    return np.dot(R_z, np.dot(R_x, R_y))


def create_masks(image):
    masks = torch.zeros((1, height, width), dtype=torch.uint8)
    return masks


def read_line_file(filename, min_line_length=10):
    segs = []  # line segments
    # csv 파일 열어서 Line 정보 가져오기
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            segs.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    segs = np.array(segs, dtype=np.float32)
    lengths = LA.norm(segs[:, 2:] - segs[:, :2], axis=1)
    segs = segs[lengths > min_line_length]
    return segs


def normalize_segs(segs, pp, rho):
    pp = np.array([pp[0], pp[1], pp[0], pp[1]], dtype=np.float32)
    return rho * (segs - pp)


def normalize_safe_np(v, axis=-1, eps=1e-6):
    de = LA.norm(v, axis=axis, keepdims=True)
    de = np.maximum(de, eps)
    return v / de


def segs2lines_np(segs):
    ones = np.ones(len(segs))
    ones = np.expand_dims(ones, axis=-1)
    p1 = np.concatenate([segs[:, :2], ones], axis=-1)
    p2 = np.concatenate([segs[:, 2:], ones], axis=-1)
    lines = np.cross(p1, p2)
    return normalize_safe_np(lines)


def sample_segs_np(segs, num_sample, use_prob=True):
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


def sample_vert_segs_np(segs, thresh_theta=22.5):
    lines = segs2lines_np(segs)
    (a, b) = lines[:, 0], lines[:, 1]
    theta = np.arctan2(np.abs(b), np.abs(a))
    thresh_theta = np.radians(thresh_theta)
    return segs[theta < thresh_theta]
def make_transform(self):
    return T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )


path = "/home/kmuvcl/source/CuTi/matterport/mp3d_planercnn_json/cached_set_test.json"
root = "/home/kmuvcl/source/CuTi/matterport/"

with open(osp.join(path)) as f:
    split = json.load(f)

scene_info = {
            "images": [],
            "poses": [],
            "intrinsics": [],
            "lines": [],
        }  # line 정보 추가
base_pose = np.array([0, 0, 0, 0, 0, 0, 1])

img_filename = []

target = {}
extra = {}

for i in tqdm(range(len(split["data"]))):
    images = []
    lines = []

    for imgnum in ["0", "1"]:
        img_name = os.path.join(
            root,
            "/".join(str(split["data"][i][imgnum]["file_name"]).split("/")[6:]),
        )
        print("img:",img_name)
        img_filename.append(img_name)
        image = cv2.imread(img_name)
        image = image[:, :, ::-1]  # convert to rgb'

        org_image = image
        org_h, org_w = image.shape[0], image.shape[1]
        org_sz = np.array([org_h, org_w])

        image = cv2.resize(image, dsize=(512, 512))
        input_sz = np.array([512, 512])

        ratio_x = float(512) / float(org_w)
        ratio_y = float(512) / float(org_h)

        pp = (org_w / 2, org_h / 2)
        rho = 2.0 / np.minimum(org_w, org_h)

        line_name = img_name.split("/")
        line_name[8] = img_name.split("/")[8].split(".")[0] + "_line.csv"
        line_name = "/".join(line_name)

        try:
            pd.read_csv(line_name)
        except FileNotFoundError:
            img = cv2.imread(img_name, 0)
            lsd = cv2.createLineSegmentDetector(0)
            lines = lsd.detect(img)[0]
            drawn_img = lsd.drawSegments(img, lines)

            Line_image_name = img_name.split("/")
            csv_file_name = img_name.split("/")
            Line_image_name[8] = (
            img_name.split("/")[8].split(".")[0]
            + "_line."
            + img_name.split("/")[8].split(".")[1]
            )
            csv_file_name[8] = img_name.split("/")[8].split(".")[0] + "_line.csv"
            Line_image_name = "/".join(Line_image_name)
            csv_file_name = "/".join(csv_file_name)
            print("cv2:",Line_image_name)
            cv2.imwrite(Line_image_name, drawn_img)
            data_df = pd.DataFrame(lines[0][0])
            data_df = data_df.T
            for i in range(1, len(lines)):
                data_df2 = pd.DataFrame(lines[i][0])
                data_df2 = data_df2.T
                data_df = pd.concat([data_df, data_df2])
            print("cv3:", csv_file_name)
            data_df.to_csv(csv_file_name, index=False, header=None)


        org_segs = read_line_file(line_name, 10)
        num_segs = len(org_segs)

        segs = normalize_segs(org_segs, pp=pp, rho=rho)

        sampled_segs, line_mask = sample_segs_np(segs, 512)
        sampled_lines = segs2lines_np(sampled_segs)

        image = np.ascontiguousarray(image)

        target["segs"] = (
            torch.from_numpy(np.ascontiguousarray(sampled_segs))
            .contiguous()
            .float()
        )
        target["lines"] = (
            torch.from_numpy(np.ascontiguousarray(sampled_lines))
            .contiguous()
            .float()
        )
        extra["lines"] = target["lines"].clone()

data = make_transform(image, extra, target)