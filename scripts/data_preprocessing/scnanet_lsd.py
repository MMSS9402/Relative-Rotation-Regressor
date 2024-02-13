import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp
from pathlib import Path
import pickle
import json
import pandas as pd
import h5py

from tqdm import tqdm

from line_detector import LSD

import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp
import pickle
import csv
import numpy.linalg as LA



conf = {
    "feature": {
        "linetr": {
            "min_length": 16,
            "token_distance": 8,
            "max_tokens": 21,
            "remove_borders": 1,
            "max_sublines": 250,
            "thred_reprojected": 3,
            "thred_angdiff": 2,
            "min_overlap_ratio": 0.3
        },
    }
}


split = "valid"
rootdir = "/home/kmuvcl/dataset/data/processed_data"
resize = (512,512)

def get_angles(lines):
    line_exists = (len(lines) > 0)
    if not line_exists:
        angles = []
        return angles

    sp = lines[:,0]
    ep = lines[:,1]
    angles = np.arctan2((ep[:,0]-sp[:,0]), (ep[:,1]-sp[:,1]))
    for i, angle in enumerate(angles):
        if angle < 0:
            angles[i] += np.pi
    angles = np.asarray([np.cos(2*angles), np.sin(2*angles)]).T
    return angles

def change_cv2_T_np(klines_cv):
    klines_sp, klines_ep, length, angle = [], [], [], []

    for line in klines_cv:
        sp_x = line.startPointX
        sp_y = line.startPointY
        ep_x = line.endPointX
        ep_y = line.endPointY
        kline_sp = []
        if sp_x < ep_x:
            kline_sp = [sp_x, sp_y]
            kline_ep = [ep_x, ep_y]
        else:
            kline_sp = [ep_x, ep_y]
            kline_ep = [sp_x, sp_y]
        
        # linelength = math.sqrt((kline_ep[0]-kline_sp[0])**2 +(kline_ep[1]-kline_sp[1])**2)
        linelength = line.lineLength*(2**line.octave)
        
        klines_sp.append(kline_sp)
        klines_ep.append(kline_ep)
        length.append(linelength)
        
    klines_sp = np.asarray(klines_sp)
    klines_ep = np.asarray(klines_ep)
    klines = np.stack((klines_sp, klines_ep), axis=1)
    length = np.asarray(length)
    angles = get_angles(klines)
    return {'klines':klines, 'length_klines':length, 'angles': angles}

def remove_borders(lines, border: int, height: int, width: int, valid_mask_given=None):
    """ Removes keylines too close to the border """
    klines = lines['klines']
    valid_mask_h0 = (klines[:, 0, 0] >= border) & (klines[:, 0, 0] < (width - border))
    valid_mask_w0 = (klines[:, 0, 1] >= border) & (klines[:, 0, 1] < (height - border))

    valid_mask_h1 = (klines[:, 1, 0] >= border) & (klines[:, 1, 0] < (width - border))
    valid_mask_w1 = (klines[:, 1, 1] >= border) & (klines[:, 1, 1] < (height - border))

    valid_mask0 = valid_mask_h0 & valid_mask_w0
    valid_mask1 = valid_mask_h1 & valid_mask_w1
    valid_mask = valid_mask0 & valid_mask1

    eps=0.001
    klines[:,:,0] = klines[:,:,0].clip(max = width-eps-border)
    klines[:,:,1] = klines[:,:,1].clip(max = height-eps-border)
    
    if isinstance(valid_mask_given, np.ndarray):
        sp = np.floor(klines[:,0]).astype(int)
        ep = np.floor(klines[:,1]).astype(int)
        valid_mask_given = ((valid_mask_given[sp[:,1], sp[:,0]] + valid_mask_given[ep[:,1], ep[:,0]])).astype(bool)
        valid_mask = valid_mask & valid_mask_given
        
    lines = {k: v[valid_mask] for k, v in lines.items()}
        
    return lines

def filter_by_length(lines, min_length, max_sublines):
    length = lines['length_klines']
    valid_idx = length>min_length
    
    klines = lines['klines'][valid_idx]
    length = lines['length_klines'][valid_idx]
    angles = lines['angles'][valid_idx]

    # re-ordering by line length
    index = np.argsort(length)
    index = index[::-1]
    klines = klines[index[:max_sublines]]
    length = length[index[:max_sublines]]

    angles = get_angles(klines)
    return {'klines':klines}#, 'length_klines':length, 'angles': angles}

def load_h5py_to_dict(file_path):
    with h5py.File(file_path, 'r') as f:
        return {key: torch.tensor(f[key][:]) for key in f.keys()}


def main():
    linedetector = LSD(conf['feature']['linetr'])
    # dirs = np.genfromtxt(f"{rootdir}/scannetv2_{split}.txt", dtype=str)
    # filelist = sum([sorted(glob.glob(f"{rootdir}/{d}/*.png")) for d in dirs], [])
    # filelist = sorted(glob.glob(f"{rootdir}/*/*_0.png"))
    filelist = glob.glob(f"{rootdir}/*_0.png")
    for path in tqdm(filelist):
        line_path = path.replace(".png","_line.h5py")
        np_path = path.replace("color.png","vanish.npz")
        if Path(line_path).exists():
            continue
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image.astype('float32'),(resize[0], resize[1]))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray.astype('uint8'), (resize[0], resize[1]))
        klns_cv = linedetector.detect(gray)
        klines =change_cv2_T_np(klns_cv)
        height, width = resize[1],resize[0]
        border = 0

        try:
            klines = remove_borders(klines, border, height, width, None)
        except IndexError:
            # klines['klines'] = torch.zeros(256,2,2).tolist()
            print(path)
            os.remove(path)
            os.remove(np_path)
            continue


        klines = filter_by_length(klines, 16, 250)
        num_klines = len(klines['klines'])
        # if num_klines ==250:
        #     pass
        # elif(num_klines>250):
        #     print(num_klines)
        # else:
        #     print("num_klines :",num_klines)
        
        with h5py.File(line_path, 'a') as fd:
            try:
                for k, v in klines.items():
                    fd.create_dataset(k, data=v.tolist())
            except OSError as error:
                    raise error
            except ValueError:
                # print(line_path)
                pass
    
    
if __name__ == '__main__':
    main()
    