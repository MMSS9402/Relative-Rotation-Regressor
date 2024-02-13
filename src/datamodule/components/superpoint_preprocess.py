from superpoint import SuperPoint
from linetr_utils.line_detector import LSD
from linetr_utils.line_process import preprocess, line_tokenizer
from linetr_utils.util_lines import find_line_matches, calculate_line_overlaps, conv_fixed_size
import torch
import gc
import os
import os.path as osp

from datetime import date
import json
import numpy as np
# import numpy.linalg as LA
import torch.linalg
from tqdm import tqdm
import pickle
import h5py
import cv2
import csv


conf = {
    "data": {
        "name": "gcl_synth",
        "image_path": "./assets/dataset/raw_images",
        "output_path": "./assets/dataset/dataset_h5",
        "image_type": "*.jpg",
        "resize": (640, 480),
        "visualize": True,
        "n_iters": 1,
        "choose_worker": 0,
        "nWorkers": 1
    },
    "augmentation": {
        "num": 1,
        "photometric": {
            "enable": True,
            "primitives": [
                "random_brightness", "random_contrast", "additive_speckle_noise",
                "additive_gaussian_noise", "additive_shade", "motion_blur"
            ],
            "params": {
                "random_brightness": {"max_abs_change": 50},
                "random_contrast": {"strength_range": [0.3, 1.5]},
                "additive_gaussian_noise": {"stddev_range": [0, 10]},
                "additive_speckle_noise": {"prob_range": [0, 0.0035]},
                "additive_shade": {
                    "transparency_range": [-0.5, 0.5],
                    "kernel_size_range": [100, 150]
                },
                "motion_blur": {"max_kernel_size": 3}
            }
        },
        "homographic": {
            "enable": True,
            "params": {
                "perspective": True,
                "scaling": True,
                "translation": True,
                "rotation": True,
                "patch_ratio": 0.85,
                "perspective_amplitude_x": 0.2,
                "perspective_amplitude_y": 0.2,
                "scaling_amplitude": 0.2,
                "max_angle": 1.0472,
                "allow_artifacts": True
            },
            "valid_border_margin": 3
        }
    },
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
        "superpoint": {
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "remove_borders": 4,
            "max_keypoints": 256
        }
    }
}

path = "/home/kmuvcl/source/oldCuTi/CuTi/matterport/mp3d_planercnn_json/cached_set_moon_test_vp.json"
root = "/home/kmuvcl/source/oldCuTi/CuTi/matterport/"


def read_data(path, device, resize):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if resize[0] != -1:
        image = cv2.resize(image.astype('float32'), (resize[0], resize[1]))
        gray = cv2.resize(gray.astype('uint8'), (resize[0], resize[1]))
    gray_torch = torch.from_numpy(gray/255.).float()[None, None].to(device)
    
    return image, gray, gray_torch

def read_line_file(filename: str, min_line_length=10):
        segs = []  # line segments

        with open(filename, "r") as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                segs.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        segs = np.array(segs, dtype=np.float32)
        lengths = LA.norm(segs[:, 2:] - segs[:, :2], axis=1)
        segs = segs[lengths > min_line_length]
        return segs
    
def to_cpu(tensor_dict):
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            tensor_dict[key] = value.cpu()
    return tensor_dict

def save_tensor_dict_to_csv(tensor_dict, csv_filename):
    # 딕셔너리의 모든 텐서 값을 numpy 배열로 변환합니다.
    numpy_dict = {key: value.cpu().numpy() for key, value in tensor_dict.items()}

    # CSV 파일로 저장합니다.
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 헤더를 작성합니다.
        writer.writerow(numpy_dict.keys())
        
        # 행을 작성합니다.
        for row in zip(*numpy_dict.values()):
            writer.writerow(row)


def main():
    device = torch.device('cuda')
    model = SuperPoint(conf['feature']['superpoint']).to(device).eval()
    linedetector = LSD(conf['feature']['linetr'])
    
    with open(path) as file:
        split = json.load(file)
        for i in tqdm(range(len(split['data']))):
            del split['data'][str(i)]['gt_corrs']
            del split['data'][str(i)]['0']['annotations']
            del split['data'][str(i)]['1']['annotations']

    original_basepath = "/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v4_sep20"
    data_path = "/home/kmuvcl/source/oldCuTi/CuTi/matterport"

    resize = (640,480)
    klines_dict = {}
    for i in tqdm(range(len(split['data']))):
        vps = []
        images_list = []
        line_list = []
        for img_idx in ["0", "1"]:
            img_path = split['data'][str(i)][img_idx]["file_name"].replace(original_basepath, data_path)
            line_path = img_path.replace(".png", "_sp_line.h5py",)
            images_list.append(img_path)
            line_list.append(line_path)
        images = []
        grays = []
        for j in range(2):
            images.append(cv2.imread(images_list[j], cv2.IMREAD_COLOR))

        images[0] = cv2.resize(images[0].astype('float32'), (resize[0], resize[1]))
        images[1] = cv2.resize(images[1].astype('float32'), (resize[0], resize[1]))
        grays.append(cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY))
        grays.append(cv2.cvtColor(images[1], cv2.COLOR_RGB2GRAY))
        
        grays[0] = cv2.resize(grays[0].astype('uint8'), (resize[0], resize[1]))
        grays[1] = cv2.resize(grays[1].astype('uint8'), (resize[0], resize[1]))

        image0_torch = torch.from_numpy(grays[0]/255.).float()[None, None].to(device)
        image1_torch = torch.from_numpy(grays[1]/255.).float()[None, None].to(device)
        
        height, width = image_shape = grays[0].shape
        
        with torch.no_grad():
            pred0 = model({'image': image0_torch})
            pred1 = model({'image': image1_torch})
        valid_mask0 = np.ones_like(grays[0])
        valid_mask1 = np.ones_like(grays[1])  
        
        klns0_cv = linedetector.detect(grays[0])
        klns1_cv = linedetector.detect(grays[1])
        
        try:
            klines0 = preprocess(klns0_cv, image_shape, pred0, mask=valid_mask0, conf=conf['feature']['linetr'])   ## TODO: torch vs. np. 잘 정리하기, tokenizer 다시 정리
            klines1 = preprocess(klns1_cv, image_shape, pred1, mask=valid_mask1, conf=conf['feature']['linetr'])
        except:
            print("line preprocess break")
            break
        klines0 = conv_fixed_size(klines0, conf, func_token=line_tokenizer, pred_sp=pred0)
        klines1 = conv_fixed_size(klines1, conf, func_token=line_tokenizer, pred_sp=pred1)
        # klns0 = klines0['sublines'].reshape(-1, 2, 2).cpu().numpy()
        # klns1 = klines1['sublines'].reshape(-1, 2, 2).cpu().numpy()
        
        keys_l = ['klines', 'sublines', 'angle_sublines','pnt_sublines', 'desc_sublines', \
                'score_sublines', 'resp_sublines', 'mask_sublines', 'num_klns', 'mat_klines2sublines', 'num_slns']
            # klines, resp, angle, pnt, desc, score, mask
        # klines0 = {k:v[0].tolist() for k,v in klines0.items() if k in keys_l}
        # klines1 = {k:v[0] for k,v in klines1.items() if k in keys_l}
        # print(line_list[0])
        # print(line_list[1])

        with h5py.File(line_list[0], 'a') as fd:
            try:
                for k, v in klines0.items():
                    fd.create_dataset(k, data=v.tolist())
            except OSError as error:
                    raise error
            except ValueError:
                print(line_list[0])
                pass
        with h5py.File(line_list[1], 'a') as fd:
            try:
                for k, v in klines1.items():
                    fd.create_dataset(k, data=v.tolist())
            except OSError as error:
                    raise error
            except ValueError:
                print(line_list[1])
                pass
        # with open('dict.json', 'w') as f:
        #     json.dump(klines0, f)
            
        # with open('dict.pkl', 'wb') as f:
        #     pickle.dump(klines0, f)
        # with open(line_list[0], 'w') as file:
        #     writer = csv.writer(file)
        #     for k, v in klines0.items():
        #         writer.writerow([k, v])
            
        # with open(line_list[1], 'w') as file:
        #     writer = csv.writer(file)
        #     for k, v in klines1.items():
        #         writer.writerow([k, v])
    

if __name__ == '__main__':
    main()