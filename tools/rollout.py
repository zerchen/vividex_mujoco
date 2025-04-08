# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import glob, yaml, os, imageio, cv2, shutil
from stable_baselines3 import PPO
from argparse import ArgumentParser
from glob import glob

import _init_paths
from hand_imitation.env.models import asset_abspath
from hand_imitation.env.environments.create_env import create_env
from hand_imitation.env.environments.gym_wrapper import GymWrapper

os.environ['MUJOCO_GL']='egl'

"""
PLEASE DOWNLOAD AND UNZIP THE PRE-TRAINED AGENTS BEFORE RUNNING THIS
SEE: https://github.com/facebookresearch/DexMan#pre-trained-policies
"""

parser = ArgumentParser(description="Example code for loading pre-trained policies")
parser.add_argument('--save_folder', '-e', default='pretrained_agents/hammer_use1/', help="Save folder containing agent checkpoint/config")
parser.add_argument('--task', '-t', default=None, help="Save folder containing agent checkpoint/config")
parser.add_argument('--render', action="store_true", help="Supply flag to render mp4")


def render(writer, physics, AA=2, height=256, width=256):
    if writer is None:
        return
    img = physics.render(camera_id=0, height=height * AA, width=width * AA)
    writer.append_data(cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA))


def render_point_clouds(physics, height=256, width=256):
    if writer is None:
        return
    img = physics.render(camera_id=0, height=height, width=width)


def rollout(save_folder):
    # get experiment config
    config =  yaml.safe_load(open(os.path.join(save_folder, 'exp_config.yaml'), 'r'))
    rollout_video_dir = os.path.join(save_folder, 'rollout_videos')
    os.makedirs(rollout_video_dir, exist_ok=True)
    policy = PPO.load(os.path.join(save_folder, 'models', 'model.zip'))
    
    # build environment and load policy
    env_name = config['params']['env']['name']
    if args.task is not None:
        env_name = args.task
    task_kwargs = config['params']['env']['task_kwargs']
    env_kwargs = config['params']['env']['env_kwargs']

    if len(env_name.split('-')) <= 2:
        object_category = env_name.split('-')[0]
        object_name = env_name.split('-')[1]
        traj_root = asset_abspath('objects/trajectories')
        traj_candidate_list = glob(f'{traj_root}/{object_category}/*{object_name}*')
        for idx, _ in enumerate(traj_candidate_list):
            traj_candidate_list[idx] = traj_candidate_list[idx].split('/')[-1].split('.')[0]
    else:
        traj_candidate_list = [env_name]
        
    for traj in traj_candidate_list:
        env = create_env(name=traj, robot_name='adroit', task_kwargs=task_kwargs, environment_kwargs=env_kwargs)
        env = GymWrapper(env)
        writer = imageio.get_writer(f'{rollout_video_dir}/{traj}.mp4', fps=25)

        # rollout the policy and print total reward
        s, done, total_reward = env.reset(), False, 0
        render(writer, env.wrapped.physics)
        while not done:
            action, _ = policy.predict(s['state'], deterministic=True)
            render(writer, env.wrapped.physics)
            s, r, done, info = env.step(action)
            total_reward += r
        writer.close()

        print(f'{traj}:', total_reward)
        if task_kwargs['only_goal']:
            print('imitate contact:', info['imitate_contact'])
            print('imitate lift:', info['imitate_lift'])
        else:
            print('imitate object error:', info['imitate_obj_err'])
            print('imitate object success:', info['imitate_obj_success'])
            print('imitate hand error:', info['imitate_hand_err'])
            print('imitate hand success:', info['imitate_hand_success'], '\n')


if __name__ == "__main__":
    args = parser.parse_args()
    rollout(args.save_folder)
