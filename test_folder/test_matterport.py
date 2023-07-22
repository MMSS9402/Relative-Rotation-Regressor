import numpy as np
import argparse
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from test_data.factory_test import dataset_factory

from lietorch import SE3
# from src.geom.losses import geodesic_loss

# network
# from src.model import ViTEss

# DDP training

from tqdm import tqdm
import os

from src.models.cuti import build
from config import cfg

def eval_camera(predictions):
    acc_threshold = {
        "tran": 1.0,
        "rot": 30,
    }  # threshold for translation and rotation error to say prediction is correct.

    pred_tran = np.vstack(predictions["camera"]["preds"]["tran"])
    pred_rot = np.vstack(predictions["camera"]["preds"]["rot"])

    gt_tran = np.vstack(predictions["camera"]["gts"]["tran"])
    gt_rot = np.vstack(predictions["camera"]["gts"]["rot"])
    # predictions["camera"]["preds"]["tran"] = np.array(predictions["camera"]["preds"]["tran"])
    # predictions["camera"]["preds"]["rot"] = np.array(predictions["camera"]["preds"]["rot"])

    # predictions["camera"]["gts"]["tran"] = predictions["camera"]["gts"]["tran"].cpu().numpy()
    # predictions["camera"]["gts"]["rot"] =  predictions["camera"]["gts"]["rot"].cpu().numpy()

    # print("prediction__________",predictions)
    # print("preds_trans________",predictions["camera"]["preds"]["tran"])
    # print("preds_rotation________",predictions["camera"]["gts"]["rot"])
    # pred_tran = np.vstack(predictions["camera"]["preds"]["tran"])
    # pred_rot = np.vstack(predictions["camera"]["preds"]["rot"])

    # gt_tran = np.vstack(predictions["camera"]["gts"]["tran"])
    # gt_rot = np.vstack(predictions["camera"]["gts"]["rot"])

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

if __name__ == '__main__':
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
    
    output_folder = 'matterport_test'
    
    print('performing evaluation on %s set using model %s' % (output_folder, args.ckpt))

    
    try:
        os.makedirs(os.path.join('output', args.exp, output_folder))
    except:
        pass
    
    
    db = dataset_factory(
        ["matterport"],
        datapath="/home/kmuvcl/source/CuTi/matterport",
        subepoch=0,
        is_training=False,
        gpu = 0,
        streetlearn_interiornet_type = None,
        use_mini_dataset = False
    )
    model = build(cfg)

    ckpt = args.ckpt
    state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(ckpt)['model'].items()])
    model.load_state_dict(state_dict,False)
    model = model.cuda().eval()


    test_loader = DataLoader(
    db, batch_size=cfg.SOLVER.BATCH_SIZE, num_workers=1, shuffle=False
    )

    DEPTH_SCALE =5
    predictions = {'camera': {'preds': {'tran': [], 'rot': []}, 'gts': {'tran': [], 'rot': []}}}
    loss_list =[]


    with tqdm(test_loader, unit="batch") as tepoch:
        for i_batch, item in enumerate(tepoch):
            images, poses, intrinsics, lines = [x.to("cuda") for x in item]
            Ps = SE3(poses) 
            Gs = SE3.IdentityLike(Ps)
            Ps_out = SE3(Ps.data.clone())
            try:
                model(images,lines,Gs)
            except RuntimeError:
                continue
            with torch.no_grad():
                poses_est = model(images, lines, Gs)
            #print(poses.shape)
            #print(poses[0][1])
            for i in range(6):
                gt_poses = poses[i][1].data.cpu().numpy()
                #print(gt_poses)
                gt_copy = np.copy(gt_poses)
                predictions['camera']['gts']['tran'].append(gt_poses[:3])
                gt_rotation = gt_poses[3:]
                temp = gt_rotation[0]
                gt_rotation[0] = gt_rotation[3]
                gt_rotation[3] = temp
                if gt_rotation[0] < 0: # normalize quaternions to have positive "W" (equivalent)
                    gt_rotation[0] *= -1
                    gt_rotation[1] *= -1
                    gt_rotation[2] *= -1
                    gt_rotation[3] *= -1
                predictions['camera']['gts']['rot'].append(gt_rotation)
                
                preds = poses_est[0][i][1].data.cpu().numpy()
                
                pr_copy = np.copy(preds)
                
                preds[3] = pr_copy[6] # swap 3 & 6, we used W last; want W first in quat
                preds[6] = pr_copy[3]
                preds[:3] = preds[:3] * DEPTH_SCALE 
                
                predictions['camera']['preds']['tran'].append(preds[:3])
                predictions['camera']['preds']['rot'].append(preds[3:])
                #print(predictions['camera']['gts']['tran'])
                #print(predictions['camera']['gts']['rot'])
            #print(predictions)  
                
    camera_metrics = eval_camera(predictions)

    for k in camera_metrics:
        print(k, camera_metrics[k])
