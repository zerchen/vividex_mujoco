import os
import time
import cv2
import torch
import copy
import random
import numpy as np
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from utils.pc_utils import trans_pcs

class TrajectoryDataset(Dataset):
    def __init__(self, db):
        self.dataset_name = db.name
        self.point_cs = db.point_cs
        self.point_num = db.point_num
        self.point_noise = db.point_noise
        self.db = db.data

    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, index):
        sample_data = copy.deepcopy(self.db[index])

        obs = torch.from_numpy(sample_data['obs'])
        random_idx = np.random.permutation(2048)[:self.point_num]
        if self.point_noise:
            pcs = torch.from_numpy(trans_pcs(sample_data['pcs'][random_idx] + np.random.normal(0, 0.01, size=(self.point_num, 3)).astype(np.float32), sample_data['hand_pos'], sample_data['hand_rot'], sample_data['final_goal'], self.point_cs))
        else:
            pcs = torch.from_numpy(trans_pcs(sample_data['pcs'][random_idx], sample_data['hand_pos'], sample_data['hand_rot'], sample_data['final_goal'], self.point_cs))
        act = torch.from_numpy(sample_data['act'])

        input_iter = dict(observations=obs, point_clouds=pcs)
        label_iter = dict(actions=act)

        return input_iter, label_iter
