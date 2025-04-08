import torch
import torch.nn as nn
import numpy as np
import time
from torch.nn import functional as F
from config import cfg


class PointNetEncoder(nn.Module):
    def __init__(self, point_cs):
        super(PointNetEncoder, self).__init__()
        if point_cs == "world":
            self.point_input_dim = 3
        elif point_cs == "hand":
            self.point_input_dim = 21
        elif point_cs == "target":
            self.point_input_dim = 6
        elif point_cs == "all":
            self.point_input_dim = 24

        self.conv1 = torch.nn.Conv1d(self.point_input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 1024, 1)

        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 256)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn4 = torch.nn.BatchNorm1d(1024)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class model(nn.Module):
    def __init__(self, cfg):
        super(model, self).__init__()
        self.cfg = cfg
        self.policy_fc0 = nn.Linear(self.cfg.obs_dim + self.cfg.pc_latent_dim, self.cfg.mlp_dim[0])
        self.policy_fc1 = nn.Linear(self.cfg.mlp_dim[0], self.cfg.mlp_dim[1])
        self.policy_fc2 = nn.Linear(self.cfg.mlp_dim[1], self.cfg.act_dim)
        self.point_network = PointNetEncoder(self.cfg.point_cs)

        self.loss_l2 = torch.nn.MSELoss()
    
    def forward(self, inputs, targets=None, mode='train'):
        if mode == 'train':
            loss = {}
            pointnet_latent = self.point_network(inputs['point_clouds'])
            latent = torch.cat((inputs['observations'], pointnet_latent), axis=1)

            policy_latent = torch.tanh(self.policy_fc0(latent))
            policy_latent = torch.tanh(self.policy_fc1(policy_latent))
            actions = self.policy_fc2(policy_latent)

            loss['act_reg'] = self.loss_l2(actions, targets['actions'])
            return loss, actions
        else:
            with torch.no_grad():
                pointnet_latent = self.point_network(inputs['point_clouds'])
                latent = torch.cat((inputs['observations'], pointnet_latent), axis=1)

                policy_latent = torch.tanh(self.policy_fc0(latent))
                policy_latent = torch.tanh(self.policy_fc1(policy_latent))
                actions = self.policy_fc2(policy_latent)

            return actions


def get_model(cfg):
    visual_model = model(cfg)
    return visual_model


if __name__ == '__main__':
    model = get_model(cfg)
