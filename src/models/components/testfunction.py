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
from typing import Callable
from lietorch import SE3


class Testcamera(Callable):
    def __init__(self, Reference):
        self.Reference_tr = Reference.translation
        self.Reference_rot = Reference.rotation
        
    def __call__(self, gt, pose_preds,predictions):
        
        predictions = predictions
        
        # preds = pose_preds[0][0][1].data.cpu().numpy()
        preds = pose_preds.data.cpu().numpy()
        print("preds", pose_preds.data)
        #print(preds.shape)
        preds = preds[:,1,:]
        #print(preds.shape)
        pr_copy = np.copy(preds)
        
        preds[:,3] = pr_copy[:,6] # swap 3 & 6, we used W last; want W first in quat
        preds[:,6] = pr_copy[:,3]
        preds[:,:3] = preds[:,:3] * 5
        
        predictions['camera']['preds']['tran'].append(preds[:,:3])
        predictions['camera']['preds']['rot'].append(preds[:,3:])
        #print("________________-")
        #print(gt.shape)
        gt = gt[:,1,:]#.squeeze(0)
        gt = gt.data.cpu().numpy()
        #print(gt.shape)
        gt_copy = np.copy(gt)
        gt[:,3] = gt_copy[:,6] # swap 3 & 6, we used W last; want W first in quat
        gt[:,6] = gt_copy[:,3]
        gt[:,:3] = gt[:,:3] * 5
        gt_tran = gt[:,:3]
        gt_rotation = gt[:,3:]
        batch_size = gt_rotation.shape[0]
        for i in range(batch_size):
            if gt_rotation[i,0] < 0: # normalize quaternions to have positive "W" (equivalent)
                gt_rotation[i,0] *= -1
                gt_rotation[i,1] *= -1
                gt_rotation[i,2] *= -1
                gt_rotation[i,3] *= -1
        predictions['camera']['gts']['tran'].append(gt_tran)
        predictions['camera']['gts']['rot'].append(gt_rotation)
        
        
        return predictions