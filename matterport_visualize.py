import torch
import gc
import os
import os.path as osp
import argparse
from datetime import date
import json
import numpy as np
import numpy.linalg as LA
import torch.linalg
from tqdm import tqdm
import pickle
import h5py
import cv2
import csv

import torch.nn.functional as F

import matplotlib as mpl
import matplotlib.pyplot as plt

path = "/home/kmuvcl/source/oldCuTi/CuTi/matterport/mp3d_planercnn_json/cached_set_moon_val_vp.json"
root = "/home/kmuvcl/source/oldCuTi/CuTi/matterport"

def line_label(target_vp1,target_vp2,target_vp3,target_lines):
    

    target_lines = torch.tensor(target_lines)
    target_vp1 = torch.tensor(target_vp1) # [bs, 3]
    target_vp2 = torch.tensor(target_vp2)# [bs, 3]
    target_vp3 = torch.tensor(target_vp3)# [bs, 3]

    target_vp1 = target_vp1.unsqueeze(0)
    target_vp2 = target_vp2.unsqueeze(0)
    target_vp3 = target_vp3.unsqueeze(0)
    thresh_line_pos = np.cos(np.radians(88.0), dtype=np.float32)
    thresh_line_neg = np.cos(np.radians(85.0), dtype=np.float32)
    with torch.no_grad():

        cos_sim_zvp = F.cosine_similarity(target_lines, target_vp1, dim=-1).abs()
        cos_sim_hvp1 = F.cosine_similarity(target_lines, target_vp2, dim=-1).abs()
        cos_sim_hvp2 = F.cosine_similarity(target_lines, target_vp3, dim=-1).abs()
        cos_sim_zvp = cos_sim_zvp.unsqueeze(-1)
        cos_sim_hvp1 = cos_sim_hvp1.unsqueeze(-1)
        cos_sim_hvp2 = cos_sim_hvp2.unsqueeze(-1)
        ones = torch.ones(250,1)
        zeros = torch.zeros(250,1)

        cos_class_1 = torch.where(cos_sim_zvp < thresh_line_pos, ones, zeros)
        cos_class_2 = torch.where(cos_sim_hvp1 < thresh_line_pos, ones, zeros)
        cos_class_3 = torch.where(cos_sim_hvp2 < thresh_line_pos, ones, zeros)


        mask_zvp = torch.where(torch.gt(cos_class_1, thresh_line_pos) &
                        torch.lt(cos_class_1, thresh_line_neg),  
                        zeros, ones)
        mask_hvp1 = torch.where(torch.gt(cos_class_2, thresh_line_pos) &
                        torch.lt(cos_class_2, thresh_line_neg),  
                        zeros, ones)
        mask_hvp2 = torch.where(torch.gt(cos_class_3, thresh_line_pos) &
                        torch.lt(cos_class_3, thresh_line_neg),  
                        zeros, ones)
        
        cos_sim = torch.where(cos_class_1==1, 1, 0) + torch.where(cos_class_2==1, 2, 0) + torch.where(cos_class_3==1, 3, 0)

        overlaps = cos_sim >= 4
        
        if overlaps.any():
            values, indices = torch.stack([cos_sim_zvp, cos_sim_hvp1, cos_sim_hvp2], dim=-1)[overlaps].min(dim=-1)
        
            cos_sim[overlaps] = indices + 1

        cos_class_1 = cos_sim == 1
        cos_class_2 = cos_sim == 2
        cos_class_3 = cos_sim == 3

        cos_class_1 = cos_class_1.float()
        cos_class_2 = cos_class_2.float()
        cos_class_3 = cos_class_3.float()

    return cos_class_1, cos_class_2,cos_class_3,mask_zvp,mask_hvp1,mask_hvp2

def load_h5py_to_dict(file_path):
    with h5py.File(file_path, 'r') as f:
        return {key: torch.tensor(f[key][:]) for key in f.keys()}

def coordinate_yup(segs, org_h):
    H = np.array([0, org_h, 0, org_h])
    segs[:, 1] = -segs[:, 1]
    segs[:, 3] = -segs[:, 3]
    return (H + segs)

def normalize_segs(lines, pp, rho=517.97):
    pp = np.array([pp[0], pp[1], pp[0], pp[1]], dtype=np.float32)
    return (lines - pp)/rho

def normalize_safe_np(v, axis=-1, eps=1e-6):
    de = LA.norm(v, axis=axis, keepdims=True)
    de = np.maximum(de, eps)
    return v / de
   
def segs2lines_np(segs):
    ones = np.ones(len(segs))
    ones = np.expand_dims(ones, axis=-1)
    p1 = np.concatenate([segs[:, :2], -ones], axis=-1)
    p2 = np.concatenate([segs[:, 2:], -ones], axis=-1)
    lines = np.cross(p1, p2)
    return normalize_safe_np(lines)

def main():
    with open(os.path.join(path)) as file:
        split = json.load(file)

    original_basepath = "/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v4_sep20"
    for data in tqdm(split["data"].values()):
        images = []
        h5py_path = []
        vps = []
        for img_idx in ["0", "1"]:
            img_path = data[img_idx]["file_name"].replace(original_basepath, root)
            print(img_path)
            images.append(img_path)
            h5py_path.append(load_h5py_to_dict(img_path.replace(".png", "_sp_line.h5py",)))
            vp1 = data[img_idx]['vp1']
            vp2 = data[img_idx]['vp2']
            vp3 = data[img_idx]['vp3']

            gt_vps = np.array([vp1, vp2, vp3])
            vps.append(gt_vps)

        rho = 517.97
        pp = (320,240)
        # angle_sublines0 = h5py_path[0]['angle_sublines'][0].float()
        # angles0 = h5py_path[0]['angles'][0].float()
        # desc_sublines0 = h5py_path[0]['desc_sublines'][0].float()
        # length_klines0 = h5py_path[0]['length_klines'][0].float()
        # mask_sublines0 = h5py_path[0]['mask_sublines'][0].float()
        # mat_klines2sublines0 = h5py_path[0]['mat_klines2sublines'][0].float()
        # num_klns0 = h5py_path[0]['num_klns'][0].float()
        # num_slns0 = h5py_path[0]['num_slns'][0].float()
        # pnt_sublines0 = h5py_path[0]['pnt_sublines'][0].float()
        # resp_sublines0 = h5py_path[0]['resp_sublines'][0].float()
        # score_sublines0 = h5py_path[0]['score_sublines'][0].float()
        klines0 = h5py_path[0]['klines'][0].float()
        sublines0 = h5py_path[0]['sublines'][0].float()
        segs0 = sublines0.reshape(250,-1).numpy()
        num_segs = 250
        img0_vps = vps[0]
        lines0 = np.copy(segs0)
        lines0 = coordinate_yup(lines0,480)
        lines0 = normalize_segs(lines0, pp=pp, rho=rho)
        normal0 = np.copy(lines0)
        normal0 = segs2lines_np(normal0)


        # angle_sublines1 = h5py_path[1]['angle_sublines'][0].float()
        # angles1 = h5py_path[1]['angles'][0].float()
        # desc_sublines1 = h5py_path[1]['despp = (images.shape[-1] / 2, images.shape[-2] / 2) k_sublines'][0].float()
        # mat_klines2sublines1 = h5py_path[1]['mat_klines2sublines'][0].float()
        # num_klns1 = h5py_path[1]['num_klns'][0].float()
        # num_slns1 = h5py_path[1]['num_slns'][0].float()
        # pnt_sublines1 = h5py_path[1]['pnt_sublines'][0].float()
        # resp_sublines1 = h5py_path[1]['resp_sublines'][0].float()
        # score_sublines1 = h5py_path[1]['score_sublines'][0].float()
        klines1 = h5py_path[1]['klines'][0].float()
        sublines1 = h5py_path[1]['sublines'][0].float()
        segs1 = sublines1.reshape(250,-1).numpy()
        img1_vps = vps[1]
        lines1 = np.copy(segs1)
        lines1 = coordinate_yup(lines1,480)
        lines1 = normalize_segs(lines1, pp=pp, rho=rho)
        normal1 = np.copy(lines1)
        normal1 = segs2lines_np(normal1)

        img0 = cv2.imread(images[0],cv2.IMREAD_COLOR)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1 = cv2.imread(images[1],cv2.IMREAD_COLOR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        class_zvp,class_hvp1,class_hvp2,mask_zvp,mask_hvp1,mask_hvp2 = line_label(img0_vps[0],img0_vps[1],img0_vps[2],normal0)
        class_zvp0 = class_zvp*mask_zvp
        class_hvp0_1 = class_hvp1*mask_hvp1
        class_hvp0_2 = class_hvp2*mask_hvp2

        class_zvp,class_hvp1,class_hvp2,mask_zvp,mask_hvp1,mask_hvp2 = line_label(img1_vps[0],img1_vps[1],img1_vps[2],normal1)
        class_zvp1 = class_zvp*mask_zvp
        class_hvp1_1 = class_hvp1*mask_hvp1
        class_hvp1_2 = class_hvp2*mask_hvp2
         
        fig_output_dir = "/home/kmuvcl/source/CuTi/matterport_vis/rgb"
        filename0 = images[0][48:]
        filename1 = images[1][48:]

        save_path0 = osp.join(fig_output_dir, filename0+'.jpg')
        save_path1 = osp.join(fig_output_dir, filename1+'.jpg')

        directory0 = os.path.dirname(save_path0)
        directory1 = os.path.dirname(save_path1)
        
        if not os.path.exists(directory0):
            os.makedirs(directory0)
        if not os.path.exists(directory1):
            os.makedirs(directory1)

        
        plt.figure(figsize=(4, 3))
        plt.imshow(img0, extent=[0, 640, 480, 0])                                 
        plt.xlim(0, 640)
        plt.ylim(480,0)
        #plt.axis('off')
        plt.savefig(osp.join(fig_output_dir, filename0+'.jpg'),  
                    pad_inches=0, bbox_inches='tight')
        plt.close('all')

        plt.figure(figsize=(4, 3))
        plt.imshow(img1, extent=[0, 640, 480, 0])                                 
        plt.xlim(0, 640)
        plt.ylim(480,0)
        #plt.axis('off')
        plt.savefig(osp.join(fig_output_dir, filename1+'.jpg'),  
                    pad_inches=0, bbox_inches='tight')
        plt.close('all')

        plt.figure(figsize=(4,3))
        #                 plt.title('zenith vp lines')
        plt.imshow(img0, extent=[0, 640, 480, 0])
        for i in range(num_segs):
            plt.plot(
                (segs0[i, 0], segs0[i, 2]),
                (segs0[i, 1], segs0[i, 3]),
                c="r",
                alpha=1.0,
            )  
        plt.xlim(0, 640)
        plt.ylim(480,0)
        #plt.axis("off")
        plt.savefig(
            osp.join(fig_output_dir, filename0 + "_lines_segment.jpg"),
            pad_inches=0,
            bbox_inches="tight",
        )

        plt.figure(figsize=(4,3))
        #                 plt.title('zenith vp lines')
        plt.imshow(img1, extent=[0, 640, 480, 0])
        for i in range(num_segs):
            plt.plot(
                (segs1[i, 0], segs1[i, 2]),
                (segs1[i, 1], segs1[i, 3]),
                c="r",
                alpha=1.0,
            )  
        plt.xlim(0, 640)
        plt.ylim(480,0)
        #plt.axis("off")
        plt.savefig(
            osp.join(fig_output_dir, filename1 + "_lines_segment.jpg"),
            pad_inches=0,
            bbox_inches="tight",
        )
        

        plt.figure(figsize=(4,3))
        plt.imshow(img0, extent=[-320/517.97,320/517.97, -240/517.97, 240/517.97])                 
        for i in range(num_segs):
            plt.plot(
                (lines0[i, 0], lines0[i, 2]),
                (lines0[i, 1], lines0[i, 3]),
                c="r",
                alpha=1.0,
            ) 
        #plt.plot((0, pred_vp1[0]), (0, pred_vp1[1]), 'g-', alpha=1.0)  
        plt.xlim(-320/517.97,320/517.97)
        plt.ylim(-240/517.97,240/517.97)
        #plt.axis('off')
        plt.savefig(osp.join(fig_output_dir, filename0+'_normalize_segs.jpg'),  
                    pad_inches=0, bbox_inches='tight')
        plt.close('all')

        plt.figure(figsize=(4,3))
        plt.imshow(img1, extent=[-320/517.97,320/517.97, -240/517.97, 240/517.97])                 
        for i in range(num_segs):
            plt.plot(
                (lines1[i, 0], lines1[i, 2]),
                (lines1[i, 1], lines1[i, 3]),
                c="r",
                alpha=1.0,
            ) 
        #plt.plot((0, pred_vp1[0]), (0, pred_vp1[1]), 'g-', alpha=1.0)  
        plt.xlim(-320/517.97,320/517.97)
        plt.ylim(-240/517.97,240/517.97)
        #plt.axis('off')
        plt.savefig(osp.join(fig_output_dir, filename1+'_normalize_segs.jpg'),  
                    pad_inches=0, bbox_inches='tight')
        plt.close('all')

        plt.figure(figsize=(4,3))
        plt.imshow(img0, extent=[-320/517.97,320/517.97, -240/517.97, 240/517.97])                 
        plt.plot((0, img0_vps[0][0]), (0, img0_vps[0][1]), 'r-', alpha=1.0)
        plt.plot((0, img0_vps[1][0]), (0, img0_vps[1][1]), 'g-', alpha=1.0)
        plt.plot((0, img0_vps[2][0]), (0, img0_vps[2][1]), 'b-', alpha=1.0)
        #plt.plot((0, pred_vp1[0]), (0, pred_vp1[1]), 'g-', alpha=1.0)  
        plt.xlim(-320/517.97,320/517.97)
        plt.ylim(-240/517.97,240/517.97)
        #plt.axis('off')
        plt.savefig(osp.join(fig_output_dir, filename0+'_gt_vps.jpg'),  
                    pad_inches=0, bbox_inches='tight')
        plt.close('all')

        plt.figure(figsize=(4,3))
        plt.imshow(img1, extent=[-320/517.97,320/517.97, -240/517.97, 240/517.97])                 
        plt.plot((0, img1_vps[0][0]), (0, img1_vps[0][1]), 'r-', alpha=1.0)
        plt.plot((0, img1_vps[1][0]), (0, img1_vps[1][1]), 'g-', alpha=1.0)
        plt.plot((0, img1_vps[2][0]), (0, img1_vps[2][1]), 'b-', alpha=1.0)
        #plt.plot((0, pred_vp1[0]), (0, pred_vp1[1]), 'g-', alpha=1.0)  
        plt.xlim(-320/517.97,320/517.97)
        plt.ylim(-240/517.97,240/517.97)
        #plt.axis('off')
        plt.savefig(osp.join(fig_output_dir, filename1+'_gt_vps.jpg'),  
                    pad_inches=0, bbox_inches='tight')
        plt.close('all')



        plt.figure(figsize=(4,3))
        plt.imshow(img0, extent=[-320/517.97,320/517.97, -240/517.97, 240/517.97])
        for i in range(num_segs):
            if class_zvp0[i] == 1 and class_hvp0_1[i] == 0 and class_hvp0_2[i] == 0:
                plt.plot(
                    (lines0[i, 0], lines0[i, 2]),
                    (lines0[i, 1], lines0[i, 3]),
                    c="r",
                    alpha=1.0,
            ) 
        for i in range(num_segs):
            if class_hvp0_1[i] == 1 and class_zvp0[i] == 0 and class_hvp0_2[i] == 0:
                plt.plot(
                (lines0[i, 0], lines0[i, 2]),
                (lines0[i, 1], lines0[i, 3]),
                c="g",
                alpha=1.0,
            )    
        for i in range(num_segs):
            if class_hvp0_2[i] == 1 and class_zvp0[i] == 0 and class_hvp0_1[i] == 0:
                plt.plot(
                (lines0[i, 0], lines0[i, 2]),
                (lines0[i, 1], lines0[i, 3]),
                c="b",
                alpha=1.0,
            ) 
        plt.xlim(-320/517.97,320/517.97)
        plt.ylim(-240/517.97,240/517.97)
        #plt.axis("off")
        plt.savefig(
            osp.join(fig_output_dir, filename0 + "mask_class.jpg"),
            pad_inches=0,
            bbox_inches="tight",
        )
        plt.close("all")

        plt.figure(figsize=(4,3))
        plt.imshow(img1, extent=[-320/517.97,320/517.97, -240/517.97, 240/517.97])
        for i in range(num_segs):
            if class_zvp1[i] == 1 and class_hvp1_1[i] == 0 and class_hvp1_2[i] == 0:
                plt.plot(
                    (lines1[i, 0], lines1[i, 2]),
                    (lines1[i, 1], lines1[i, 3]),
                    c="r",
                    alpha=1.0,
            ) 
        for i in range(num_segs):
            if class_hvp1_1[i] == 1 and class_zvp1[i] == 0 and class_hvp1_2[i] == 0:
                plt.plot(
                (lines1[i, 0], lines1[i, 2]),
                (lines1[i, 1], lines1[i, 3]),
                c="g",
                alpha=1.0,
            )    
        for i in range(num_segs):
            if class_hvp1_2[i] == 1 and class_zvp1[i] == 0 and class_hvp1_1[i] == 0:
                plt.plot(
                (lines1[i, 0], lines1[i, 2]),
                (lines1[i, 1], lines1[i, 3]),
                c="b",
                alpha=1.0,
            ) 
        plt.xlim(-320/517.97,320/517.97)
        plt.ylim(-240/517.97,240/517.97)
        #plt.axis("off")
        plt.savefig(
            osp.join(fig_output_dir, filename1 + "mask_class.jpg"),
            pad_inches=0,
            bbox_inches="tight",
        )
        plt.close("all")

  

            
if __name__ == '__main__':
    main()