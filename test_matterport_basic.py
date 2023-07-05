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

from cuti import build
from config import cfg

from collections import OrderedDict
import pickle
import json
import numpy.linalg as LA
import csv
from lietorch import SE3

import glm

DEPTH_SCALE = 5

def normalize_segs( lines, pp, rho):
        pp = np.array([pp[0], pp[1], pp[0], pp[1]], dtype=np.float32)
        return rho * (lines - pp)

def sample_segs_np( segs, num_sample, use_prob=True):
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

def normalize_safe_np( v, axis=-1, eps=1e-6):
    de = LA.norm(v, axis=axis, keepdims=True)
    de = np.maximum(de, eps)
    return v / de

def segs2lines_np( segs):
    ones = np.ones(len(segs))
    ones = np.expand_dims(ones, axis=-1)
    p1 = np.concatenate([segs[:, :2], ones], axis=-1)
    p2 = np.concatenate([segs[:, 2:], ones], axis=-1)
    lines = np.cross(p1, p2)
    return normalize_safe_np(lines)

def read_line_file(filename, min_line_length=10):
    segs = []  # line segments
    # csv 파일 열어서 Line 정보 가져오기
    
    with open(str(filename), "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            segs.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    segs = np.array(segs, dtype=np.float32)
    lengths = LA.norm(segs[:, 2:] - segs[:, :2], axis=1)
    segs = segs[lengths > min_line_length]
    return segs
def coordinate_yup(segs,org_h):
        H = np.array([0,org_h,0,org_h])
        segs[:,1] = -segs[:,1]
        segs[:,3] = -segs[:,3]
        return(H+segs)

def eval_camera(predictions):
    acc_threshold = {
        "tran": 1.0,
        "rot": 30,
    }  # threshold for translation and rotation error to say prediction is correct.

    pred_tran = np.vstack(predictions["camera"]["preds"]["tran"])
    pred_rot = np.vstack(predictions["camera"]["preds"]["rot"])

    gt_tran = np.vstack(predictions["camera"]["gts"]["tran"])
    gt_rot = np.vstack(predictions["camera"]["gts"]["rot"])

    # VP_rot = np.vstack(predictions['VP_rot']['preds'])
    # VP_gt = np.vstack(predictions['VP_rot']['gt'])

    top1_error = {
        "tran": np.linalg.norm(gt_tran - pred_tran, axis=1),
        "rot": 2 * np.arccos(np.clip(np.abs(np.sum(np.multiply(pred_rot, gt_rot), axis=1)), -1.0, 1.0)) * 180 / np.pi,
        #"vp_rot" : 2 * np.arccos(np.clip(np.abs(np.sum(np.multiply(VP_rot, VP_gt ), axis=1)), -1.0, 1.0)) * 180 / np.pi
    }
    top1_accuracy = {
        "tran": (top1_error["tran"] < acc_threshold["tran"]).sum()
        / len(top1_error["tran"]),
        "rot": (top1_error["rot"] < acc_threshold["rot"]).sum()
        / len(top1_error["rot"]),
    }
    camera_metrics = {
        f"top1 T err < {acc_threshold['tran']}": top1_accuracy["tran"] * 100,
        f"top1 R err < {acc_threshold['rot']}": top1_accuracy["rot"] * 100,
        #f"VP" : top1_accuracy["vp_rot"] * 100,
        f"T mean err": np.mean(top1_error["tran"]),
        f"R mean err": np.mean(top1_error["rot"]),
        f"T median err": np.median(top1_error["tran"]),
        f"R median err": np.median(top1_error["rot"]),
        # f"VP_rot" : VP_rot,
        # f"VP_gt" : VP_gt,
        # f"VP_error" : np.sum(np.abs(VP_rot - VP_gt))/len(VP_rot)
    }
    
    gt_mags = {"tran": np.linalg.norm(gt_tran, axis=1), "rot": 2 * np.arccos(gt_rot[:,0]) * 180 / np.pi}

    tran_graph = np.stack([gt_mags['tran'], top1_error['tran']],axis=1)
    tran_graph_name = os.path.join('output', args.exp, output_folder, 'gt_translation_magnitude_vs_error.csv')
    np.savetxt(tran_graph_name, tran_graph, delimiter=',', fmt='%1.5f')

    rot_graph = np.stack([gt_mags['rot'], top1_error['rot']],axis=1)
    rot_graph_name = os.path.join('output', args.exp, output_folder, 'gt_rotation_magnitude_vs_error.csv')
    np.savetxt(rot_graph_name, rot_graph, delimiter=',', fmt='%1.5f')

    # vprot_graph = np.stack([VP_rot, VP_gt],axis=1)
    # rot_graph_name = os.path.join('output', args.exp, output_folder, 'VProt_vs_gt_rot.csv')
    # np.savetxt(rot_graph_name, rot_graph, delimiter=',', fmt='%1.5f')
    
    return camera_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--datapath")
    parser.add_argument("--weights")
    parser.add_argument("--image_size", default=[480,640])
    parser.add_argument("--exp")
    parser.add_argument("--ckpt")
    parser.add_argument('--gamma', type=float, default=0.9)    

    # model
    parser.add_argument('--no_pos_encoding', action='store_true')
    parser.add_argument('--noess', action='store_true')
    parser.add_argument('--cross_features', action='store_true')
    parser.add_argument('--use_single_softmax', action='store_true')  
    parser.add_argument('--l1_pos_encoding', action='store_true')
    parser.add_argument('--fusion_transformer', action="store_true", default=False)
    parser.add_argument('--fc_hidden_size', type=int, default=512)
    parser.add_argument('--pool_size', type=int, default=60)
    parser.add_argument('--transformer_depth', type=int, default=6)

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    with open(os.path.join(args.datapath, 'mp3d_planercnn_json/cached_set_test.json')) as f:
        test_split = json.load(f)

    dset = test_split
    output_folder = 'matterport_test'

    print('performing evaluation on %s set using model %s' % (output_folder, args.ckpt))

    try:
        os.makedirs(os.path.join('output', args.exp, output_folder))
    except:
        pass

    model = build(cfg)
    #print("model________",model)
    state_dict = OrderedDict([
        (k.replace("module.", ""), v) for (k, v) in torch.load(args.ckpt)['model'].items()])
    model.load_state_dict(state_dict,False)
    model = model.cuda().eval()
    
    train_val = ''
    predictions = {'VP_rot':{'preds':[],'gt':[]},'camera': {'preds': {'tran': [], 'rot': []}, 'gts': {'tran': [], 'rot': []}}}

    for i in tqdm(range(len(dset['data']))):
        images = []
    
        lines = []
        for imgnum in ['0', '1']:
            img_name = os.path.join(args.datapath, '/'.join(str(dset['data'][i][imgnum]['file_name']).split('/')[6:]))
            images.append(cv2.imread(img_name))
            #print(img_name)
            line_name = img_name.split("/")
            line_name[9] = img_name.split("/")[9].split(".")[0] + "_line.csv"
            line_name = "/".join(line_name)
            lines.append(read_line_file(line_name,10))

        pp = (640 / 2, 480 / 2)
        rho = 2.0 / np.minimum(640,480)
        lines[0] = coordinate_yup(lines[0],480)
        lines[0] = normalize_segs(lines[0], pp=pp, rho=rho)
        lines[0] = sample_segs_np(lines[0], 512)
        lines[0] = segs2lines_np(lines[0])
        lines[1] = coordinate_yup(lines[1],480)
        lines[1] = normalize_segs(lines[1], pp=pp, rho=rho)
        lines[1] = sample_segs_np(lines[1], 512)
        lines[1] = segs2lines_np(lines[1])

        lines = np.array(lines)
        lines = torch.from_numpy(lines).float()
        lines = lines.unsqueeze(0).cuda()
        #print(lines.shape)

        images = np.stack(images).astype(np.float32)
        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)
        images = F.interpolate(images, size=[480,640])
        images = images.unsqueeze(0).cuda()
        #print(images.shape)
        intrinsics = np.stack([np.array([[517.97, 517.97, 320, 240], [517.97, 517.97, 320, 240]])]).astype(np.float32)
        intrinsics = torch.from_numpy(intrinsics).cuda()

        base_pose = np.array([0,0,0,0,0,0,1])
        poses = np.vstack([base_pose, base_pose]).astype(np.float32)
        poses = torch.from_numpy(poses).unsqueeze(0).cuda()
        Gs = SE3(poses)
                    
        with torch.no_grad():
            poses_est = model(images, lines, Gs)
        #print("++++++",poses_est2)
        #preds2 = poses_est2[0][0][1].data.cpu().numpy()
        # pr_copy2 = np.copy(preds2)
        # preds2[3] = pr_copy2[6] # swap 3 & 6, we used W last; want W first in quat
        # preds2[6] = pr_copy2[3]
        # preds2[:3] = preds2[:3] * DEPTH_SCALE 
        # VP_rot = preds2[3:]
        # #VP_rot = glm.eulerAngles(VP_rot)


        #predictions['VP_rot']['preds'].append(VP_rot)


        predictions['camera']['gts']['tran'].append(dset['data'][i]['rel_pose']['position'])
        gt_rotation = dset['data'][i]['rel_pose']['rotation']
        
        #print("relpose",dset['data'][i]['rel_pose'])
        
        if gt_rotation[0] < 0: # normalize quaternions to have positive "W" (equivalent)
            gt_rotation[0] *= -1
            gt_rotation[1] *= -1
            gt_rotation[2] *= -1
            gt_rotation[3] *= -1
        predictions['camera']['gts']['rot'].append(gt_rotation)
        #gt_rot = glm.eulerAngles(gt_rotation)
        #gt_rot = gt_rotation
        #predictions['VP_rot']['gt'].append(gt_rot)

        preds = poses_est[0][0][1].data.cpu().numpy()    
        pr_copy = np.copy(preds)

        # undo preprocessing we made during training, for evaluation
        preds[3] = pr_copy[6] # swap 3 & 6, we used W last; want W first in quat
        preds[6] = pr_copy[3]
        preds[:3] = preds[:3] * DEPTH_SCALE 
        # print("VPROT",VP_rot)
        # print("gt",gt_rot)
        #print('error',np.abs(VP_rot-gt_rot))
        predictions['camera']['preds']['tran'].append(preds[:3])
        predictions['camera']['preds']['rot'].append(preds[3:])

        # print("gt_rotation________",predictions['camera']['gts']['rot'])
        # print("gt_translation________",predictions['camera']['gts']['tran'])
        # print("pred_rotation________",predictions['camera']['preds']['tran'])
        # print("pred_rotation________",predictions['camera']['preds']['rot'])
        
        #print(predictions)

    camera_metrics = eval_camera(predictions)
    for k in camera_metrics:
        print(k, camera_metrics[k])
    
    with open(os.path.join('output', args.exp, output_folder, 'results.txt'), 'w') as f:
        for k in camera_metrics:
            print(k, camera_metrics[k], file=f)
