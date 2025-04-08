# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import numpy as np
import glob, yaml, os, imageio, cv2, shutil
from stable_baselines3 import PPO
from argparse import ArgumentParser
from glob import glob
from dm_control import mujoco
from loguru import logger


import _init_paths
from hand_imitation.env.environments.create_env import create_env
from hand_imitation.env.environments.gym_wrapper import GymWrapper
from hand_imitation.env.models.control import _normalize_action, _denormalize_action
from hand_imitation.env.models import Environment, ObjMimicTask
from hand_imitation.env.models import TableEnv, asset_abspath, get_robot, get_object, physics_from_mjcf
from hand_imitation.env.utils.random import np_random
# from openpoints.models.backbone.pointnet import PointNetEncoder

os.environ['MUJOCO_GL']='egl'

"""
PLEASE DOWNLOAD AND UNZIP THE PRE-TRAINED AGENTS BEFORE RUNNING THIS
SEE: https://github.com/facebookresearch/DexMan#pre-trained-policies
"""

parser = ArgumentParser(description="Example code for loading pre-trained policies")
parser.add_argument('--task', '-t', default=None, help="define tasks used for training or evaluation")
parser.add_argument('--output_name', '-o', default=None, help="Save folder containing agent checkpoint/config")


class PointEncoder(nn.Module):
    def __init__(self):
        super(PointEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
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


class MLP:
    def __init__(self, hidden_sizes=(64, 64), min_log_std=-3, init_log_std=0, seed=None, use_pointnet=False):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        if use_pointnet:
            self.n = 271 + 256
        else:
            self.n = 302
        self.m = 30  # number of actions
        self.min_log_std = min_log_std
        self.use_pointnet = use_pointnet

        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Policy network
        # ------------------------
        self.model = MuNet(self.n, self.m, hidden_sizes)
        # make weights small
        for param in list(self.model.parameters())[-2:]:  # only last layer
            param.data = 1e-2 * param.data

        self.log_std = Variable(torch.ones(self.m) * init_log_std, requires_grad=True)
        self.trainable_params = list(self.model.parameters()) + [self.log_std]

        if self.use_pointnet:
            # self.pointnet = PointNetEncoder(3, feature_transform=False)
            self.pointnet = PointEncoder()

        # Easy access variables
        # -------------------------
        self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(self.n), requires_grad=False)

        self.model.cuda()
        if self.use_pointnet:
            self.pointnet.cuda()

    # Utility functions
    # ============================================
    def get_param_values(self):
        params = np.concatenate([p.contiguous().view(-1).cpu().data.numpy() for p in self.trainable_params])
        return params.copy()

    def set_param_values(self, new_params):
        #print(f'cuda before set param: {next(self.model.parameters()).is_cuda}')
        current_idx = 0
        for idx, param in enumerate(self.trainable_params):
            vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
            vals = vals.reshape(self.param_shapes[idx])
            if param.is_cuda:
                param.data = torch.from_numpy(vals).float().cuda()
            else:
                param.data = torch.from_numpy(vals).float()
            current_idx += self.param_sizes[idx]
        # clip std at minimum value
        self.trainable_params[-1].data = torch.clamp(self.trainable_params[-1], self.min_log_std).data
        # update log_std_val for sampling
        self.log_std_val = np.float64(self.log_std.cpu().data.numpy().ravel())

    # Main functions
    # ============================================
    def get_action(self, observation, pc=None):
        with torch.no_grad():
            if pc is None:
                self.model.eval()
                o = np.float32(observation.reshape(1, -1))
                self.obs_var.data = torch.from_numpy(o).cuda()
            else:
                self.model.eval()
                self.pointnet.eval()
                o = torch.from_numpy(np.float32(observation.reshape(1, -1))).cuda()
                pc = torch.from_numpy(np.float32(pc.reshape(1, -1, 3))).cuda()
                # pc_feat = self.pointnet.forward_cls_feat(pc)
                pc_feat = self.pointnet(pc)
                self.obs_var.data = torch.cat((o, pc_feat), axis=1)
        mean = self.model(self.obs_var).data.cpu().numpy().ravel()
        noise = np.exp(self.log_std_val) * np.random.randn(self.m)
        action = mean + noise
        return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]

    def predict(self, obs, pc):
        # out = self.pointnet.forward_cls_feat(pc)
        out = self.pointnet(pc)
        out = torch.cat((obs, out), axis=1)
        out = self.model(out)
        return out


class MuNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64,64), in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        super(MuNet, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)

        self.fc0 = nn.Linear(obs_dim, hidden_sizes[0])
        self.fc1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc2 = nn.Linear(hidden_sizes[1], act_dim)

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # store native scales that can be used for resets
        self.transformations = dict(in_shift=in_shift, in_scale=in_scale, out_shift=out_shift, out_scale=out_scale)
        self.in_shift  = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale  = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)
        self.in_shift  = Variable(self.in_shift, requires_grad=False).cuda()
        self.in_scale  = Variable(self.in_scale, requires_grad=False).cuda()
        self.out_shift = Variable(self.out_shift, requires_grad=False).cuda()
        self.out_scale = Variable(self.out_scale, requires_grad=False).cuda()

    def forward(self, x):
        out = (x - self.in_shift)/(self.in_scale + 1e-8)
        out = torch.tanh(self.fc0(out))
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        out = out * self.out_scale + self.out_shift
        return out


class BC:
    def __init__(self, output_path, expert_paths, policy, epochs=5, batch_size=64, lr=1e-3, optimizer=None):
        self.policy = policy
        self.expert_paths = expert_paths
        self.epochs = epochs
        self.mb_size = batch_size
        self.use_pointnet = hasattr(self.policy, 'pointnet')

        self.job_dir = os.path.join(output_path, 'iterations')
        os.makedirs(self.job_dir, exist_ok=True)

        if self.use_pointnet:
            self.optimizer = torch.optim.Adam([{'params': self.policy.model.parameters(), 'lr':lr}, {'params': self.policy.pointnet.parameters(), 'lr':lr},])
        else:
            # get transformations
            observations = np.concatenate([path["observations"] for path in expert_paths])
            actions = np.concatenate([path["actions"] for path in expert_paths])
            in_shift, in_scale = np.mean(observations, axis=0), np.std(observations, axis=0)
            out_shift, out_scale = np.mean(actions, axis=0), np.std(actions, axis=0)
            # set scalings in the target policy
            self.policy.model.set_transformations(in_shift, in_scale, out_shift, out_scale)
            # set the variance of gaussian policy based on out_scale
            params = self.policy.get_param_values()
            params[-self.policy.m:] = np.log(out_scale + 1e-12)
            self.policy.set_param_values(params)
            self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=lr)

        # loss criterion is MSE for maximum likelihood estimation
        self.loss_function = torch.nn.MSELoss()

    def loss(self, obs, pc, act):
        obs_var = Variable(torch.from_numpy(obs).float(), requires_grad=False).cuda()
        pc_var = Variable(torch.from_numpy(pc).float(), requires_grad=False).cuda()
        act_var = Variable(torch.from_numpy(act).float(), requires_grad=False).cuda()
        if self.use_pointnet:
            act_hat = self.policy.predict(obs_var, pc_var)
        else:
            act_hat = self.policy.model(obs_var)
        return self.loss_function(act_hat, act_var.detach())

    def train(self):
        observations = np.concatenate([path["observations"] for path in self.expert_paths])
        pointclouds = np.concatenate([path["pcs"] for path in self.expert_paths])
        actions = np.concatenate([path["actions"] for path in self.expert_paths])

        num_samples = observations.shape[0]
        for ep in tqdm(range(self.epochs + 1)):
            for mb in range(int(num_samples / self.mb_size)):
                rand_idx = np.random.choice(num_samples, size=self.mb_size)
                obs = observations[rand_idx]
                pc = pointclouds[rand_idx]
                act = actions[rand_idx]
                self.optimizer.zero_grad()
                loss = self.loss(obs, pc, act)
                if mb % 20 == 0:
                    logger.info(f"epoch {ep}, iteration: {mb}, loss: {loss}")
                loss.backward()
                self.optimizer.step()

            if ep % 50 == 0 and ep > 0:
                pickle.dump(self.policy, open(os.path.join(self.job_dir, f'bc_{ep}.pickle'), 'wb'))


def render(writer, physics, AA=2, height=256, width=256):
    if writer is None:
        return
    img = physics.render(camera_id=0, height=height * AA, width=width * AA)
    writer.append_data(cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA))


def render_pcs(physics, object_name, cam_intr, cam_pose, sampled_points=2048):
    depth_img = np.zeros((3, 256, 256))
    physics.model.geom(f'{object_name}/target_visual').group = 3
    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.geomgroup[3] = 0
    scene_option.geomgroup[4] = 0

    for idx, camera_name in enumerate(['front', 'left', 'right']):
        camera_id = physics.model.camera(camera_name).id
        depth_img[idx] = physics.render(depth=True, camera_id=camera_id, height=256, width=256, scene_option=scene_option)

    world_points_multiview = []
    for idx in range(3):
        xmap = np.arange(256, dtype=np.float32)
        ymap = np.arange(256, dtype=np.float32)
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth_img[idx]
        points_x = (ymap - cam_intr[0, 2]) * points_z / cam_intr[0, 0]
        points_y = (xmap - cam_intr[1, 2]) * points_z / cam_intr[1, 1]
        cam_points = np.stack([points_x, points_y, points_z, np.ones_like(points_z)], axis=-1).reshape([-1, 4])

        world_points = (cam_pose[idx] @ cam_points.transpose(1, 0)).transpose(1, 0)
        world_points = world_points[:, :3] / world_points[:, [3]]
        world_points = world_points.reshape((256, 256, 3))
        world_points = world_points[np.where(depth_img[idx] > 0)].reshape((-1, 3))
        world_points_multiview.append(world_points)
    world_points_multiview = np.concatenate(world_points_multiview)
    world_points_multiview = world_points_multiview[world_points_multiview[:, 2] > 0.0012]
    num_points = world_points_multiview.shape[0]

    if num_points > 0:
        if num_points < sampled_points:
            points = np.concatenate([world_points_multiview, np.ones((sampled_points - num_points, 3))])
        else:
            random, _ = np_random()
            points = world_points_multiview[random.permutation(num_points)[:sampled_points]]
    else:
        points = np.zeros((sampled_points, 3), dtype=np.float32)

    physics.model.geom(f'{object_name}/target_visual').group = 0

    return points


def rollout():
    # get experiment config
    save_folder = os.path.join('outputs/visual', args.output_name)
    logger.add(f'{save_folder}/run.log')

    task_list = args.task.split('/')
    env_list = []
    cfg_list = []
    data_traj = []
    object_name_list = []
    env_name_list = []
    for per_task in task_list:
        config = yaml.safe_load(open(os.path.join("../datasets/dexycb_traj", f'{per_task}.yaml'), 'r'))
        cfg_list.append(config)
        env_name = config['params']['env']['name']
        task_kwargs = config['params']['env']['task_kwargs']
        env_kwargs = config['params']['env']['env_kwargs']
        env_name_list.append(env_name)

        env = create_env(name=env_name, robot_name='adroit', task_kwargs=task_kwargs, environment_kwargs=env_kwargs)
        env = GymWrapper(env)
        env._base_env._stage = 2
        env._base_env._mode = "test"
        env_list.append(env)
        object_name = env_name.split('-')[1]
        object_name_list.append(object_name)

        with open(os.path.join("../datasets/dexycb_traj/data", f"{per_task}.pkl"), 'rb') as f:
            data_traj += list(pickle.load(f).values())

    fov = env.wrapped.physics.model.camera('front').fovy[0]
    focal_scaling = (1. / np.tan(np.deg2rad(fov) / 2)) * 256 / 2.0
    focal_matrix = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]
    image_matrix = np.eye(3)
    image_matrix[0, 2] = (256 - 1) / 2.0
    image_matrix[1, 2] = (256 - 1) / 2.0
    cam_intr = image_matrix @ focal_matrix

    cam_pose = np.zeros((3, 4, 4))
    for idx, camera_name in enumerate(['front', 'left', 'right']):
        cam_pose[idx] = np.eye(4)
        cam_pose[idx, :3, :3] = env.wrapped.physics.model.camera(camera_name).mat0.reshape((3, 3)) @ np.array([[0, 1, 0],[1, 0, 0],[0, 0, -1]])
        cam_pose[idx, :3, 3] = env.wrapped.physics.model.camera(camera_name).pos
    cam_extr = np.linalg.inv(cam_pose)
    cam_mat = cam_intr[None].repeat(3, axis=0) @ cam_extr

    policy_bc = MLP(hidden_sizes=(256, 256), seed=200, init_log_std=0.0, min_log_std=-3.0, use_pointnet=True)
    bc_agent = BC(os.path.join(save_folder, 'bc_train'), data_traj, policy=policy_bc, epochs=1000, batch_size=64, lr=1e-5)
    bc_agent.train()
    # policy_bc = pickle.load(open(os.path.join(save_folder, 'bc_train', 'iterations', 'bc_1000.pickle'), 'rb'))

    count_success_list = []
    for env_idx in range(len(env_list)):
        env = env_list[env_idx]
        rollout_test_dir = os.path.join(save_folder, cfg_list[env_idx]['params']['env']['name'])
        os.makedirs(rollout_test_dir, exist_ok=True)

        count_test = 0
        count_success = 0
        for test_idx in range(100):
            writer = imageio.get_writer(f'{rollout_test_dir}/test_{count_test}.mp4', fps=25)
            s, done, total_reward = env.reset(), False, 0
            render(writer, env.wrapped.physics)

            while not done:
                obs = np.concatenate([s['hand_position'], s['hand_velocity'], s['hand_palm_pos'] - s['final_goal']])
                pcs = render_pcs(env.wrapped.physics, object_name_list[env_idx], cam_intr, cam_pose)
                _, action = policy_bc.get_action(obs, pcs)
                render(writer, env.wrapped.physics)
                s, r, done, info = env.step(action['evaluation'])
                total_reward += r
            writer.close()

            cur_env_name = env_name_list[env_idx]
            logger.info(f'{cur_env_name}: {total_reward}')
            count_success += float(info['relocate_step_obj_err'] < 0.1)
            count_test += 1
        
        success_rate = count_success / count_test
        logger.info(f'SR {cur_env_name}: {success_rate}')
        count_success_list.append(success_rate)

    mean_success_rate = np.mean(count_success_list)
    logger.info(f'overall SR: {mean_success_rate}')


if __name__ == "__main__":
    args = parser.parse_args()
    rollout()
