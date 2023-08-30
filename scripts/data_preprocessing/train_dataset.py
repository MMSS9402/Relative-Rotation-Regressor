import cv2
from tqdm import tqdm
import numpy as np
import torch
import lietorch
import os
import glob
import time
import yaml
import argparse

import torch 
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pickle
import json

from lietorch import SE3

DEPTH_SCALE = 5

def eval_camera(predictions):
    acc_threshold = {
        "tran": 1.0,
        "rot": 30,
    }  # threshold for translation and rotation error to say prediction is correct.

    gt_tran = np.vstack(predictions["camera"]["gts"]["tran"])
    gt_rot = np.vstack(predictions["camera"]["gts"]["rot"])

    
    gt_mags = {"tran": np.linalg.norm(gt_tran, axis=1), "rot": 2 * np.arccos(gt_rot[:,0]) * 180 / np.pi}

    tran_graph = np.stack([gt_mags['tran']],axis=1)
    tran_graph_name = os.path.join('/home/kmuvcl/source/oldCuTi/CuTi/logs/train_data_output', 'gt_translation_magnitude_vs_error.csv')
    np.savetxt(tran_graph_name, tran_graph, delimiter=',', fmt='%1.5f')

    rot_graph = np.stack([gt_mags['rot']],axis=1)
    rot_graph_name = os.path.join('/home/kmuvcl/source/oldCuTi/CuTi/logs/train_data_output', 'gt_rotation_magnitude_vs_error.csv')
    np.savetxt(rot_graph_name, rot_graph, delimiter=',', fmt='%1.5f')
    
    return camera_metrics

if __name__ == '__main__':
    path = "/home/kmuvcl/source/oldCuTi/CuTi/matterport/mp3d_planercnn_json/cached_set_moon_train_vp.json"
    root = "/home/kmuvcl/source/oldCuTi/CuTi/matterport/"


    with open(path) as f:
        test_split = json.load(f)

    dset = test_split
    output_folder = 'matterport_test'
    
    predictions = {'camera': {'gts': {'tran': [], 'rot': []}}}

    for i in tqdm(range(len(dset['data']))):
        print(dset['data'][i]['rel_pose']['position'])
        predictions['camera']['gts']['tran'].append(dset['data'][i]['rel_pose']['position'])
        gt_rotation = dset['data'][i]['rel_pose']['rotation']
        if gt_rotation[0] < 0: # normalize quaternions to have positive "W" (equivalent)
            gt_rotation[0] *= -1
            gt_rotation[1] *= -1
            gt_rotation[2] *= -1
            gt_rotation[3] *= -1
        predictions['camera']['gts']['rot'].append(gt_rotation)

    camera_metrics = eval_camera(predictions)
    for k in camera_metrics:
        print(k, camera_metrics[k])
