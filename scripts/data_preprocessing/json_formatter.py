import os
import os.path as osp
import argparse
from datetime import date
import json
import random
import time
from pathlib import Path
import numpy as np
import numpy.linalg as LA
import torch.linalg
from tqdm import tqdm
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import cv2
import csv

# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# import util.misc as utils
# from config import cfg

# import pandas as pd
# import glm
from pprint import pprint

path = "/home/kmuvcl/source/oldCuTi/CuTi/matterport/mp3d_planercnn_json/cached_set_test.json"
root = "/home/kmuvcl/source/oldCuTi/CuTi/matterport/"

with open(path) as f:
        split = json.load(f)


with open('test.json','w') as f:
    json.dump(split, f, ensure_ascii=False, indent=4)