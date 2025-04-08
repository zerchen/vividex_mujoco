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
from hand_imitation.env.environments.create_env import create_env
from hand_imitation.env.environments.gym_wrapper import GymWrapper
from hand_imitation.env.models.control import _normalize_action, _denormalize_action
from hand_imitation.env.models import Environment, ObjMimicTask
from hand_imitation.env.models import TableEnv, asset_abspath, get_robot, get_object, physics_from_mjcf

os.environ['MUJOCO_GL']='egl'

"""
PLEASE DOWNLOAD AND UNZIP THE PRE-TRAINED AGENTS BEFORE RUNNING THIS
SEE: https://github.com/facebookresearch/DexMan#pre-trained-policies
"""

parser = ArgumentParser(description="Example code for loading pre-trained policies")
parser.add_argument('--save_folder', '-e', default='pretrained_agents/hammer_use1/', help="Save folder containing agent checkpoint/config")
parser.add_argument('--task', '-t', default=None, help="Save folder containing agent checkpoint/config")
parser.add_argument('--render', action="store_true", help="Supply flag to render mp4")


def set_control(physics, action):
    physics.set_control(action)
    physics.named.data.qfrc_applied[:30] = physics.named.data.qfrc_bias[:30]

def step_physics(physics):
    for _ in range(10):
        physics.step()

def rollout(save_folder):
    # get experiment config
    config = yaml.safe_load(open(os.path.join(save_folder, 'exp_config.yaml'), 'r'))
    
    # build environment and load policy
    env_name = config['params']['env']['name']
    if args.task is not None:
        env_name = args.task
    task_kwargs = config['params']['env']['task_kwargs']
    env_kwargs = config['params']['env']['env_kwargs']

    rollout_video_dir = os.path.join(save_folder, env_name)
    os.makedirs(rollout_video_dir, exist_ok=True)

    env = create_env(name=env_name, robot_name='adroit', task_kwargs=task_kwargs, environment_kwargs=env_kwargs)
    env = GymWrapper(env)
    env._base_env._stage = 3
    writer = imageio.get_writer(f'{rollout_video_dir}/{env_name}.mp4', fps=25)

    for idx in range(10):
        env.reset()
        pregrasp_step = env.wrapped.task.reference_motion.cur_reference_motion['pregrasp_step']
        robot_qpos = env.wrapped.task.reference_motion.cur_reference_motion['robot_qpos']

        actions = []
        writer = imageio.get_writer(f'{rollout_video_dir}/test_{idx}.mp4')
        writer.append_data(env.wrapped.physics.render(camera_id=4))

        for step in range(pregrasp_step - 7, pregrasp_step):
            for t in np.linspace(0, 1, num=8):
                ac = (1 - t) * robot_qpos[step] + t * robot_qpos[step + 1]
                set_control(env.wrapped.physics, ac); actions.append(_normalize_action(env.wrapped.physics, ac))
                step_physics(env.wrapped.physics)
                writer.append_data(env.wrapped.physics.render(camera_id=4))

        for _ in np.linspace(0, 1, num=5):
            ac = robot_qpos[pregrasp_step]
            set_control(env.wrapped.physics, ac); actions.append(_normalize_action(env.wrapped.physics, ac))
            step_physics(env.wrapped.physics)
            writer.append_data(env.wrapped.physics.render(camera_id=4))
    
        writer.close()

if __name__ == "__main__":
    args = parser.parse_args()
    rollout(args.save_folder)
