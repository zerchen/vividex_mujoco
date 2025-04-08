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
import trimesh
import numpy as np
import glob, yaml, os, imageio, cv2, shutil
from stable_baselines3 import PPO
from argparse import ArgumentParser
from glob import glob
from dm_control import mujoco


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
parser.add_argument('--save_folder', '-e', default='pretrained_agents/hammer_use1/', help="Save folder containing agent checkpoint/config")


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


def rollout(save_folders):
    output_dir = "data_save"
    os.makedirs(output_dir, exist_ok=True)

    for save_folder in save_folders.split('--'):
        # get experiment config
        config_path = os.path.join(save_folder, 'exp_config.yaml')
        config = yaml.safe_load(open(config_path, 'r'))
        policy = PPO.load(os.path.join(save_folder, 'restore_checkpoint.zip'))
    
        # build environment and load policy
        env_name = config['params']['env']['name']
        task_kwargs = config['params']['env']['task_kwargs']
        env_kwargs = config['params']['env']['env_kwargs']

        output_config_path = os.path.join(output_dir, f'{env_name}.yaml')
        os.system(f'cp {config_path} {output_config_path}')

        rollout_video_dir = os.path.join(save_folder, 'with_visual')
        rollout_test_dir = os.path.join(save_folder, 'with_visual_test')
        os.makedirs(rollout_video_dir, exist_ok=True)
        os.makedirs(rollout_test_dir, exist_ok=True)

        env = create_env(name=env_name, robot_name='adroit', task_kwargs=task_kwargs, environment_kwargs=env_kwargs)
        env = GymWrapper(env)
        env._base_env._stage = 2
        object_name = env_name.split('-')[1]
        desired_step = 60 if task_kwargs['action'] == "relocate" else 80

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

        count = 0
        count_success_10 = 0
        count_success_5 = 0
        count_success_3 = 0
        hand_err = []
        hand_success = []
        obj_err = []
        obj_success = []
        while count < 100:
            # writer = imageio.get_writer(f'{rollout_video_dir}/train_{count}.mp4', fps=25)
            # rollout the policy and print total reward
            s, done, total_reward = env.reset(), False, 0
            step = 0
            while not done:
                action, _ = policy.predict(s['state'], deterministic=True)
                hand_pos = np.zeros((6, 3), dtype=np.float32)
                hand_rot = np.zeros((6, 9), dtype=np.float32)
                for idx, body_name in enumerate(["adroit/palm", "adroit/thdistal_point", "adroit/ffdistal_point", "adroit/mfdistal_point", "adroit/rfdistal_point", "adroit/lfdistal_point"]):
                    hand_pos[idx] = env.wrapped.physics.named.data.xpos[body_name].copy()
                    hand_rot[idx] = env.wrapped.physics.named.data.xmat[body_name].copy()
                s, r, done, info = env.step(action)
                step += 1
                total_reward += r
            # writer.close()

            if task_kwargs['action'] == "place":
                banana_mesh = trimesh.load('../hand_imitation/env/models/assets/objects/ycb/011_banana/textured.obj', process=False)
                banana_volume = banana_mesh.volume
                mug_size = np.array([0.045, 0.08]) * 1.8
                mug_mesh = trimesh.primitives.Cylinder(radius=mug_size[0], height=mug_size[1])

            count += 1
            obj_err.append(info['imitate_obj_err'])
            obj_success.append(info['imitate_obj_success'])
            hand_err.append(info['imitate_hand_err'])
            hand_success.append(info['imitate_hand_success'])
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
                if np.linalg.norm(s['obj_pos'] - s['final_goal']) < 0.1:
                    count_success_10 += 1
                    if np.linalg.norm(s['obj_pos'] - s['final_goal']) < 0.05:
                        count_success_5 += 1
                        if np.linalg.norm(s['obj_pos'] - s['final_goal']) < 0.03:
                            count_success_3 += 1

        print(f'EO for {env_name}:', sum(obj_err) / len(obj_err))
        print(f'SO {env_name}:', sum(obj_success) / len(obj_success))
        print(f'EH {env_name}:', sum(hand_err) / len(hand_err))
        print(f'SH {env_name}:', sum(hand_success) / len(hand_success))
        print(f'SR@10 for {env_name}:', count_success_10 / count)
        print(f'SR@5 for {env_name}:', count_success_5 / count)
        print(f'SR@3 for {env_name}:', count_success_3 / count)


if __name__ == "__main__":
    args = parser.parse_args()
    rollout(args.save_folder)
