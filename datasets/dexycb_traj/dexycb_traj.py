import os
from turtle import color
import numpy as np
import time
import pickle
import json
import trimesh
import torch
from tqdm import tqdm
from loguru import logger


class dexycb_traj:
    def __init__(self, video_names, point_cs, point_num, point_noise=False, complete_pc=False):
        self.name = 'dexycb_traj'
        self.video_names = video_names.split('/')
        self.point_cs = point_cs
        self.cur_dir = os.path.dirname(__file__)
        self.traj_source = os.path.join(self.cur_dir, 'data')
        self.point_num = point_num
        self.point_noise = point_noise
        self.complete_pc = complete_pc
        self.data = self.load_data()
    
    def load_data(self):
        data_traj = []
        for filename in os.listdir(self.traj_source):
            if filename.split('.')[0] in self.video_names:
                with open(os.path.join(self.traj_source, filename), 'rb') as f:
                    data_traj += list(pickle.load(f).values())
                
        data = []
        for traj_idx in range(len(data_traj)):
            for sample_idx in range(len(data_traj[traj_idx]['observations'])):
                sample = dict()
                # sample['obs'] = data_traj[traj_idx]['observations'][sample_idx]
                sample['obs'] = np.concatenate((data_traj[traj_idx]['observations'][sample_idx], (data_traj[traj_idx]['hand_pos'][sample_idx] - data_traj[traj_idx]['final_goal'][sample_idx]).reshape(-1)))
                sample['hand_pos'] = data_traj[traj_idx]['hand_pos'][sample_idx]
                sample['hand_rot'] = data_traj[traj_idx]['hand_rot'][sample_idx]
                sample['final_goal'] = data_traj[traj_idx]['final_goal'][sample_idx]
                sample['pcs'] = data_traj[traj_idx]['pcs'][sample_idx]
                sample['act'] = data_traj[traj_idx]['actions'][sample_idx]
                data.append(sample)

        return data
    
if __name__ == "__main__":
    db = dexycb_traj()
