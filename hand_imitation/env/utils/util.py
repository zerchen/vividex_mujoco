# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import functools, gym, copy
import numpy as np
from glob import glob
from omegaconf import OmegaConf

from hand_imitation.env.environments.create_env import create_env
from hand_imitation.env.environments.gym_wrapper import GymWrapper
from hand_imitation.env.models import asset_abspath
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder


class InfoCallback(BaseCallback):
    def _on_rollout_end(self) -> None:
        all_keys = {}
        for info in self.model.ep_info_buffer:
            for k, v in info.items():
                if k in ('r', 't', 'l'):
                    continue
                elif k not in all_keys:
                    all_keys[k] = []
                all_keys[k].append(v)

        for k, v in all_keys.items():
            self.model.logger.record(f'env/{k}', np.mean(v))
    
    def _on_step(self) -> bool:
        return True


class _ObsExtractor(gym.Wrapper):
    def __init__(self, env, state_keyword):
        super().__init__(env=env)
        self._state_keyword = state_keyword
        self.observation_space = env.observation_space[state_keyword]
    
    def step(self, action):
        o, r, done, info = self.env.step(action)
        return o[self._state_keyword], r, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)[self._state_keyword]
    
    def curriculum(self, stage):
        self.env.env._base_env._stage = stage


class FallbackCheckpoint(BaseCallback):
    def __init__(self, output_dir, checkpoint_freq=1, verbose=0):
        super().__init__(verbose)
        self.output_dir = output_dir
        self.checkpoint_freq = checkpoint_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.checkpoint_freq == 0 or self.n_calls <= 1:
            self.model.save(f'{self.output_dir}/restore_checkpoint')
        return True


def _env_maker(name, task_kwargs, env_kwargs, info_keywords, state_keyword):
    np.random.seed()
    env = create_env(name=name, robot_name="adroit", task_kwargs=task_kwargs, environment_kwargs=env_kwargs)
    env = GymWrapper(env)
    env = Monitor(env, info_keywords=tuple(info_keywords))
    env = _ObsExtractor(env, state_keyword)
    return env


def sample_traj(traj_name, is_eval, n_envs):
    object_category = traj_name.split('-')[0]
    object_name = traj_name.split('-')[1]
    traj_root = asset_abspath('objects/trajectories')
    traj_candidate_list = glob(f'{traj_root}/{object_category}/*{object_name}*')
    
    if object_name == "037_scissors":
        traj_train_list = traj_candidate_list[:8]
        traj_eval_list = [traj_candidate_list[0]]
    else:
        traj_train_list = traj_candidate_list[:16]
        traj_eval_list = [traj_candidate_list[0]]
    
    if is_eval:
        return traj_eval_list * (n_envs // len(traj_eval_list))
    else:
        return traj_train_list * (n_envs // len(traj_train_list))


def make_env(multi_proc, is_eval, n_envs, vid_freq, vid_length, **kwargs):
    traj_name = kwargs['name']
    if len(traj_name.split('-')) > 2:
        env_maker = functools.partial(_env_maker, **kwargs)
        if multi_proc:
            env = SubprocVecEnv([env_maker for _ in range(n_envs)])
        else:
            env = DummyVecEnv([env_maker for _ in range(n_envs)])
    else:
        traj_sampled_list = sample_traj(traj_name, is_eval, n_envs)
        env_maker_list = []
        for i in range(n_envs):
            traj_kwargs = copy.deepcopy(kwargs)
            traj_kwargs['name'] = traj_sampled_list[i].split('/')[-1].split('.')[0]
            env_maker_list.append(functools.partial(_env_maker, **traj_kwargs))

        if multi_proc:
            env = SubprocVecEnv([env_maker_list[i] for i in range(n_envs)])
        else:
            env = DummyVecEnv([env_maker_list[i] for i in range(n_envs)])

    if vid_freq is not None:
        vid_freq = max(int(vid_freq // n_envs), 1)
        trigger = lambda x: x % vid_freq == 0 or x <= 1
        env = VecVideoRecorder(env, "videos/", record_video_trigger=trigger, video_length=vid_length)
    return env


def get_warm_start(env_args):
    traj_name = env_args.name
    if len(traj_name.split('-')) > 2:
        e = _env_maker(env_args.name, env_args.task_kwargs, env_args.env_kwargs, env_args.info_keywords, 'zero_ac')
    else:
        traj_root = asset_abspath('objects/trajectories')
        traj_candidate_list = glob(f'{traj_root}/*/{traj_name}*')
        traj_sampled = np.random.choice(traj_candidate_list, size=1, replace=True)[0]
        traj_sampled_name = traj_sampled.split('/')[-1].split('.')[0]
        e = _env_maker(traj_sampled_name, env_args.task_kwargs, env_args.env_kwargs, env_args.info_keywords, 'zero_ac')

    zero_ac = e.reset().copy().astype(np.float32)
    zero_ac[1] += 0.2; zero_ac[6:] += 0.2
    return np.clip(zero_ac, -1, 1)


def make_policy_kwargs(policy_config):
    return OmegaConf.to_container(policy_config)

