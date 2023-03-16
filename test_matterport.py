import numpy as np
import wandb
from collections import OrderedDict
import argparse
import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from test_data.factory_test import dataset_factory
from config import cfg

import lietorch
from lietorch import SE3
from geom.losses import geodesic_loss
# from src.geom.losses import geodesic_loss

# network
# from src.model import ViTEss
from logger import Logger
from cuti import build

# DDP training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import random
from datetime import datetime
import os

import torchvision.models as models

from cuti import build
from config import cfg

DEPTH_SCALE = 5

def eval_camera(predictions):
    acc_threshold = {
        "tran": 1.0,
        "rot": 30,
    }  # threshold for translation and rotation error to say prediction is correct.

    pred_tran = np.vstack(predictions["camera"]["preds"]["tran"])
    pred_rot = np.vstack(predictions["camera"]["preds"]["rot"])

    gt_tran = np.vstack(predictions["camera"]["gts"]["tran"])
    gt_rot = np.vstack(predictions["camera"]["gts"]["rot"])

    top1_error = {
        "tran": np.linalg.norm(gt_tran - pred_tran, axis=1),
        "rot": 2 * np.arccos(np.clip(np.abs(np.sum(np.multiply(pred_rot, gt_rot), axis=1)), -1.0, 1.0)) * 180 / np.pi,
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
        f"T mean err": np.mean(top1_error["tran"]),
        f"R mean err": np.mean(top1_error["rot"]),
        f"T median err": np.median(top1_error["tran"]),
        f"R median err": np.median(top1_error["rot"]),
    }
    
    gt_mags = {"tran": np.linalg.norm(gt_tran, axis=1), "rot": 2 * np.arccos(gt_rot[:,0]) * 180 / np.pi}

    tran_graph = np.stack([gt_mags['tran'], top1_error['tran']],axis=1)
    tran_graph_name = os.path.join('output', args.exp, output_folder, 'gt_translation_magnitude_vs_error.csv')
    np.savetxt(tran_graph_name, tran_graph, delimiter=',', fmt='%1.5f')

    rot_graph = np.stack([gt_mags['rot'], top1_error['rot']],axis=1)
    rot_graph_name = os.path.join('output', args.exp, output_folder, 'gt_rotation_magnitude_vs_error.csv')
    np.savetxt(rot_graph_name, rot_graph, delimiter=',', fmt='%1.5f')
    
    return camera_metrics

if __name__ == '__maiin__':
    parser = argparse.ArgumentParser()
     # data
    parser.add_argument("--datapath")
    parser.add_argument("--weights")
    parser.add_argument("--image_size", default=[384,512])
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

    db = dataset_factory(
    ["matterport"],
    datapath="/home/kmuvcl/source/CuTi/matterport",
    subepoch=0,
    is_training=False,
    gpu = 0,
    streetlearn_interiornet_type = None,
    use_mini_dataset = False
    )

    test_loader = DataLoader(db, batch_size=args.batch, num_workers=1,shuffle=False)

    model = build(cfg)
    state_dict = OrderedDict([
        (k.replace("module.", ""), v) for (k, v) in torch.load(args.ckpt)['model'].items()])
    model.load_state_dict(state_dict)
    model = model.cuda().eval()

    predictions = {'camera': {'preds': {'tran': [], 'rot': []}, 'gts': {'tran': [], 'rot': []}}}
    loss_list =[]

    with tqdm(test_loader, unit="batch") as tepoch:
        for i_batch, item in enumerate(tepoch):
            images, poses, intrinsics, lines = [x.to("cuda") for x in item]
            Ps = SE3(poses) 
            Gs = SE3.IdentityLike(Ps)
            Ps_out = SE3(Ps.data.clone())
            with torch.no_grad():
                poses_est = model(images, lines, Gs)
            #print(poses)
            