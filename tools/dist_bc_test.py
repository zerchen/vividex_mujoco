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
import subprocess
import numpy as np
import argparse
import trimesh
import glob, yaml, os, imageio, cv2, shutil
import torch.backends.cudnn as cudnn
from stable_baselines3 import PPO
from argparse import ArgumentParser
from glob import glob
from dm_control import mujoco
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

import _init_paths
from hand_imitation.env.environments.create_env import create_env
from hand_imitation.env.environments.gym_wrapper import GymWrapper
from hand_imitation.env.models.control import _normalize_action, _denormalize_action
from hand_imitation.env.models import Environment, ObjMimicTask
from hand_imitation.env.models import TableEnv, asset_abspath, get_robot, get_object, physics_from_mjcf
from hand_imitation.env.utils.random import np_random
from common.utils.pc_utils import trans_pcs
# from openpoints.models.backbone.pointnet import PointNetEncoder

os.environ['MUJOCO_GL']='egl'

"""
PLEASE DOWNLOAD AND UNZIP THE PRE-TRAINED AGENTS BEFORE RUNNING THIS
SEE: https://github.com/facebookresearch/DexMan#pre-trained-policies
"""

def parse_args():
    parser = ArgumentParser(description="Example code for loading pre-trained policies")
    parser.add_argument('--cfg', '-e', required=True, type=str)
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--test_epoch', default=0, type=int)
    parser.add_argument('--unseen', action='store_true')
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args


def render(writer, physics, AA=2, height=256, width=256):
    if writer is None:
        return
    img = physics.render(camera_id=0, height=height * AA, width=width * AA)
    writer.append_data(cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA))


def render_pcs(physics, object_name, cam_intr, cam_pose, sampled_points=2048, action="relocate"):
    depth_img = np.zeros((3, 256, 256))
    physics.model.geom(f'{object_name}/target_visual').group = 3
    if action == "pour":
        for i in range(90, 112):
            physics.model.geom(i).group = 3
    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.geomgroup[3] = 0
    scene_option.geomgroup[4] = 0

    for idx, camera_name in enumerate(['front', 'left', 'right']):
        camera_id = physics.model.camera(camera_name).id
        depth_img[idx] = physics.render(depth=True, camera_id=camera_id, height=256, width=256, scene_option=scene_option)

    if action == "pour":
        for i in range(90, 112):
            physics.model.geom(i).group = 0

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


def main():
    args = parse_args()

    from config import cfg, update_config
    from base import Tester
    from utils.dist_utils import reduce_tensor
    update_config(cfg, args, mode="test")
    if args.test_epoch == 0:
        args.test_epoch = cfg.end_epoch - 1

    local_rank = args.local_rank
    device = 'cuda:%d' % local_rank
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    logger.info('Distributed Process %d, Total %d.' % (args.local_rank, world_size))

    tester = Tester(args.test_epoch)
    tester._make_model(local_rank)

    # get experiment config
    video_list = cfg.test_video.split('/') if not args.unseen else ["ycb-002_master_chef_can-20200709-subject-01-20200709_141754", "ycb-003_cracker_box-20200813-subject-02-20200813_145653", "ycb-007_tuna_fish_can-20200709-subject-01-20200709_143626", "ycb-008_pudding_box-20200813-subject-02-20200813_151204", "ycb-009_gelatin_box-20200709-subject-01-20200709_144429", "ycb-010_potted_meat_can-20200709-subject-01-20200709_144854"]
    env_list = []
    cfg_list = []
    data_traj = []
    object_name_list = []
    env_name_list = []
    for per_task in video_list:
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

    count_success_list_10 = []
    count_success_list_5 = []
    count_success_list_3 = []
    for env_idx in range(len(env_list)):
        env = env_list[env_idx]
        rollout_test_dir = os.path.join(cfg.result_dir, cfg_list[env_idx]['params']['env']['name'])
        os.makedirs(rollout_test_dir, exist_ok=True)

        count_test = 0
        count_success_10 = 0
        count_success_5 = 0
        count_success_3 = 0
        for test_idx in range(100):
            # writer = imageio.get_writer(f'{rollout_test_dir}/test_{count_test}.mp4', fps=25)
            s, done, total_reward = env.reset(), False, 0
            # render(writer, env.wrapped.physics)

            while not done:
                hand_pos = np.zeros((6, 3), dtype=np.float32)
                hand_rot = np.zeros((6, 9), dtype=np.float32)
                for idx, body_name in enumerate(["adroit/palm", "adroit/thdistal_point", "adroit/ffdistal_point", "adroit/mfdistal_point", "adroit/rfdistal_point", "adroit/lfdistal_point"]):
                    hand_pos[idx] = env.wrapped.physics.named.data.xpos[body_name].copy()
                    hand_rot[idx] = env.wrapped.physics.named.data.xmat[body_name].copy()
                obs = torch.from_numpy(np.concatenate([s['hand_position'], s['hand_velocity'], s['hand_palm_pos'] - s['final_goal'], (hand_pos - s['final_goal']).reshape(-1)]).astype(np.float32)).cuda()
                if cfg.point_noise:
                    pcs = torch.from_numpy(trans_pcs(render_pcs(env.wrapped.physics, object_name_list[env_idx], cam_intr, cam_pose, cfg.point_num, task_kwargs['action']) + np.random.normal(0, 0.01, size=(cfg.point_num, 3)).astype(np.float32), hand_pos, hand_rot, s['final_goal'], cfg.point_cs).astype(np.float32)).cuda()
                else:
                    pcs = torch.from_numpy(trans_pcs(render_pcs(env.wrapped.physics, object_name_list[env_idx], cam_intr, cam_pose, cfg.point_num, task_kwargs['action']), hand_pos, hand_rot, s['final_goal'], cfg.point_cs).astype(np.float32)).cuda()
                cur_input = {}
                cur_input["observations"] = obs.unsqueeze(0)
                cur_input["point_clouds"] = pcs.unsqueeze(0)
                action = tester.model(cur_input, mode='eval')[0].cpu().numpy()
                # render(writer, env.wrapped.physics)
                s, r, done, info = env.step(action)
                total_reward += r
            # writer.close()

            if task_kwargs['action'] == "place":
                banana_mesh = trimesh.load('../hand_imitation/env/models/assets/objects/ycb/011_banana/textured.obj', process=False)
                banana_volume = banana_mesh.volume
                mug_size = np.array([0.045, 0.08]) * 1.8
                mug_mesh = trimesh.primitives.Cylinder(radius=mug_size[0], height=mug_size[1])

            cur_env_name = env_name_list[env_idx]
            # logger.info(f'{cur_env_name}: {total_reward}')
            if task_kwargs['action'] == "place":
                banana_mat = np.eye(4)
                banana_mat[:3, :3] = env.wrapped.physics.named.data.xmat[45].reshape((3, 3))
                banana_mat[:3, 3] = env.wrapped.physics.named.data.xpos[45]
                mug_mat = np.eye(4)
                mug_mat[:3, :3] = env.wrapped.physics.named.data.xmat[48].reshape((3, 3))
                mug_mat[:3, 3] = env.wrapped.physics.named.data.xpos[48]
                banana_mesh.apply_transform(banana_mat)
                mug_mesh.apply_transform(mug_mat)
                intersect = mug_mesh.intersection([banana_mesh])
                volume = 0 if isinstance(intersect, trimesh.Scene) else intersect.volume
                percentage = volume / banana_volume
                count_success_10 += percentage
                count_success_5 += percentage
                count_success_3 += percentage
            elif task_kwargs['action'] == "pour":
                tank_size = np.array([0.15, 0.15], dtype=np.float32)
                particle_pos = env.wrapped.physics.named.data.xpos[-12:, :].copy()
                tank_pos = np.array([-0.08, -0.1], dtype=np.float32)
                upper_limit = (tank_size[:2] / 2 + tank_pos)
                lower_limit = (-tank_size[:2] / 2 + tank_pos)
                z = 0.08
                x_within = np.logical_and(particle_pos[:, 0] < upper_limit[0], particle_pos[:, 0] > lower_limit[0])
                y_within = np.logical_and(particle_pos[:, 1] < upper_limit[1], particle_pos[:, 1] > lower_limit[1])
                z_within = np.logical_and(particle_pos[:, 2] < z, particle_pos[:, 2] > 0)
                xy_within = np.logical_and(x_within, y_within)
                tank_within = np.logical_and(z_within, xy_within)
                percentage = sum(tank_within) / len(tank_within)
                count_success_10 += percentage
                count_success_5 += percentage
                count_success_3 += percentage
            else:
                count_success_10 += float(info['relocate_step_obj_err'] < 0.1)
                count_success_5 += float(info['relocate_step_obj_err'] < 0.05)
                count_success_3 += float(info['relocate_step_obj_err'] < 0.03)
            count_test += 1

        success_rate_10 = count_success_10 / count_test
        success_rate_5 = count_success_5 / count_test
        success_rate_3 = count_success_3 / count_test
        tester.logger.info(f'SR {cur_env_name}: {success_rate_10} {success_rate_5} {success_rate_3}')
        count_success_list_10.append(success_rate_10)
        count_success_list_5.append(success_rate_5)
        count_success_list_3.append(success_rate_3)

    mean_success_rate_10 = np.mean(count_success_list_10)
    mean_success_rate_5 = np.mean(count_success_list_5)
    mean_success_rate_3 = np.mean(count_success_list_3)
    tester.logger.info(f'overall SR: {mean_success_rate_10} {mean_success_rate_5} {mean_success_rate_3}')


if __name__ == "__main__":
    main()
