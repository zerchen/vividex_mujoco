#!/usr/bin/env python3

"""Train an agent from states."""
import os
import hydra
import torch
import wandb
import yaml
import _init_paths
import numpy as np

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from algos import ActorCriticPolicy
from hand_imitation.env.environments.create_env import create_env
from hand_imitation.env.environments.gym_wrapper import GymWrapper
from hand_imitation.env.utils.util import make_env, make_policy_kwargs, InfoCallback, FallbackCheckpoint, get_warm_start
from hand_imitation.env.utils.eval import make_eval_env, EvalCallback

os.environ['MUJOCO_GL']='egl'

def create_wandb_run(output_dir, wandb_cfg, job_config, run_id=None):
    try:
        job_id = HydraConfig().get().job.num
        override_dirname = HydraConfig().get().job.override_dirname
        name = f'{wandb_cfg.sweep_name_prefix}-{job_id}'
        notes = f'{override_dirname}'
    except:
        name, notes = None, None
    return wandb.init(project=wandb_cfg.project,
                        dir=output_dir,
                        config=job_config,
                        group=wandb_cfg.group,
                        sync_tensorboard=True, 
                        monitor_gym=True,
                        save_code=True,
                        name=name,
                        notes=notes,
                        id=run_id,
                        resume=run_id is not None
                  )


cfg_path = os.path.dirname(__file__)
cfg_path = os.path.join(cfg_path, 'experiments')
@hydra.main(config_path=cfg_path, config_name="config.yaml", version_base="1.2")
def train(cfg: DictConfig, resume_model=None):
    # logging
    
    cfg_yaml = OmegaConf.to_yaml(cfg)
    resume_model = cfg.resume_model
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if os.path.exists(os.path.join(output_dir, 'exp_config.yaml')):
        old_config = yaml.load(open(os.path.join(output_dir, 'exp_config.yaml'), 'r'))
        params, wandb_id = old_config['params'], old_config['wandb_id']
        run = create_wandb_run(output_dir, cfg.wandb, params, wandb_id)
        resume_model = 'restore_checkpoint.zip'
        assert os.path.exists(os.path.join(output_dir, resume_model)), 'restore_checkpoint.zip does not exist!'
    else:
        defaults = HydraConfig.get().runtime.choices
        params = yaml.safe_load(cfg_yaml)
        params['defaults'] = {k: defaults[k] for k in ('agent', 'env')}

        run = create_wandb_run(output_dir, cfg.wandb, params)
        save_dict = dict(wandb_id=run.id, params=params)
        yaml.dump(save_dict, open(os.path.join(output_dir, 'exp_config.yaml'), 'w'))
        print('Config:')
        print(cfg_yaml)
    
    # create_env(name=cfg.env.name, task_kwargs=cfg.env.task_kwargs, environment_kwargs=cfg.env.env_kwargs)
    if cfg.agent.name == 'PPO':
        # Construct the env
        total_timesteps = cfg.total_timesteps
        eval_freq = int(cfg.eval_freq // cfg.n_envs)
        save_freq = int(cfg.save_freq // cfg.n_envs)
        restore_freq = int(cfg.restore_checkpoint_freq // cfg.n_envs)
        n_steps = int(cfg.agent.params.n_steps // cfg.n_envs)
        multi_proc = bool(cfg.agent.multi_proc)
        env = make_env(multi_proc=multi_proc, is_eval=False, **cfg.env)

        if resume_model:
            model = PPO.load(resume_model, env)
            model._last_obs = None
            reset_num_timesteps = True
        else:
            model = PPO(
                            ActorCriticPolicy, 
                            env, verbose=1, 
                            tensorboard_log=f"{output_dir}/tb_logs/", 
                            n_steps=n_steps, 
                            gamma=cfg.agent.params.gamma,
                            gae_lambda=cfg.agent.params.gae_lambda,
                            learning_rate=cfg.agent.params.learning_rate,
                            ent_coef=cfg.agent.params.ent_coef,
                            vf_coef=cfg.agent.params.vf_coef,
                            clip_range=cfg.agent.params.clip_range,
                            batch_size=cfg.agent.params.batch_size,
                            n_epochs=cfg.agent.params.n_epochs,
                            policy_kwargs=make_policy_kwargs(cfg.agent.policy_kwargs)
                        )
            # initialize the agent with behavior cloning if desired
            if cfg.agent.params.warm_start_mean:
                warm_start = get_warm_start(cfg.env)
                bias = torch.from_numpy(warm_start)
                model.policy.set_action_bias(bias)
            reset_num_timesteps = True
        
        # initialize callbacks and train
        eval_env = make_eval_env(multi_proc, cfg.n_eval_envs, **cfg.env)
        eval_callback = EvalCallback(output_dir, eval_freq, eval_env)
        restore_callback = FallbackCheckpoint(output_dir, restore_freq)
        log_info = InfoCallback()
        checkpoint = CheckpointCallback(save_freq=save_freq, save_path=f'{output_dir}/logs/', name_prefix='rl_models')
        wandb = WandbCallback(model_save_path=f"{output_dir}/models/", verbose=2)
        train_logger = configure(f'{output_dir}/logs/', ["stdout", "log"])
        model.set_logger(train_logger)
        return model.learn(
                            total_timesteps=total_timesteps,
                            callback=[log_info, eval_callback, checkpoint, restore_callback, wandb],
                            reset_num_timesteps=reset_num_timesteps
                          )

    else:
        raise NotImplementedError
    wandb.finish()



if __name__ == '__main__':
    # Load the config
    train()
