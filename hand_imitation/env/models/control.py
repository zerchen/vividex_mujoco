# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import copy, collections
import dm_env
from dm_env import specs
from dm_control.rl import control
from .rewards import ObjectMimic
from .reference import HandObjectReferenceMotion
from hand_imitation.env.utils.random import np_random


def _denormalize_action(physics, action):
    ac_min, ac_max = physics.ctrl_range.T
    ac_mid = 0.5 * (ac_max + ac_min)
    ac_range = 0.5 * (ac_max - ac_min)
    return np.clip(action, -1, 1) * ac_range + ac_mid


def _normalize_action(physics, action):
    ac_min, ac_max = physics.ctrl_range.T
    ac_mid = 0.5 * (ac_max + ac_min)
    ac_range = 0.5 * (ac_max - ac_min)
    return np.clip((action - ac_mid) / ac_range, -1, 1)


def set_control(physics, action):
    physics.set_control(action)
    physics.named.data.qfrc_applied[:30] = physics.named.data.qfrc_bias[:30]


def step_physics(physics):
    for _ in range(10):
        physics.step()


class Environment(control.Environment):
    def __init__(self, physics, task, default_camera_id=0, **kwargs):
        self._default_camera_id = default_camera_id
        self._stage = 0
        self._mode = "train"
        super().__init__(physics, task, **kwargs)

    def get_state(self):
        return dict(physics=self.physics.get_state(),task=self.task.get_state())
    
    def set_state(self, state):
        self.physics.set_state(state['physics'])
        self.task.set_state(state['task'])
    
    def reset(self):
        """Starts a new episode and returns the first `TimeStep`."""
        self._reset_next_step = False
        self._step_count = 0
        with self._physics.reset_context():
          self._task.initialize_episode(self._physics, stage=self._stage, mode=self._mode)

        observation = self._task.get_observation(self._physics)
        if self._flat_observation:
          observation = flatten_observation(observation)

        return dm_env.TimeStep(step_type=dm_env.StepType.FIRST, reward=None, discount=None, observation=observation)

    @property
    def flat_obs(self):
        return self._flat_observation
    
    @property
    def default_camera_id(self):
        return self._default_camera_id


class Task(control.Task):
    def __init__(self, reward_fns, reward_weights=None, random=None):
        # initialize 
        if not isinstance(random, np.random.RandomState):
            # random = np.random.RandomState(random)
            random, _ = np_random(random)
        self._random = random
        self._info = {}

        # store reward functions and weighting terms
        self._step_count = 0
        self._reward_fns = copy.deepcopy(reward_fns)
        reward_wgts = [1.0 for _ in self._reward_fns] if reward_weights is None else reward_weights
        self._reward_wgts = copy.deepcopy(reward_wgts)

    @property
    def random(self):
        """Task-specific `numpy.random.RandomState` instance."""
        return self._random

    def action_spec(self, physics):
        """Returns a `BoundedArraySpec` matching the `physics` actuators."""
        return specs.BoundedArray((physics.adim,), np.float32, -1, 1)

    def initialize_episode(self, physics):
        """ Sets the state of the environment at the start of each episode.
            Called by `control.Environment` at the start of each episode *within*
            `physics.reset_context()` (see the documentation for `base.Physics`)

        Args:
            physics: An instance of `mujoco.Physics`.
        """
        # initialize info dict and rewards
        self._info = {}
        self._step_count = 0
        self.initialize_rewards(physics)
    
    def initialize_rewards(self, physics):
        """ Initializes reward function objects with necessarily data/objects in task
        
        Args:
            physics: An instance of `mujoco.Physics`
        """
        for reward_fn in self._reward_fns:
            reward_fn.initialize_rewards(self, physics)

    def before_step(self, action, physics):
        """Sets the control signal for the actuators to values in `action`."""
        self._step_count += 1
        action = _denormalize_action(physics, action)
        physics.set_control(action)

    def after_step(self, physics):
        """Called immediately after environment step: no-op by default"""

    def get_observation(self, physics):
        """Returns a default observation of current physics state."""
        obs = collections.OrderedDict()
        obs['qpos'] = physics.data.qpos.astype(np.float32).copy()
        obs['qvel'] = physics.data.qvel.astype(np.float32).copy()
        obs['obj_pos'] = physics.data.xpos[45].copy()
        obs['obj_mat'] = physics.data.xmat[45].copy()
        obs['hand_pos'] = physics.data.site_xpos[[0, 10, 14, 18, 23, 27]].copy()
        obs['hand_mat'] = physics.data.site_xmat[[0, 10, 14, 18, 23, 27]].copy()
        motor_joints = physics.data.qpos[:physics.adim]
        obs['zero_ac'] = _normalize_action(physics, motor_joints)
        obs['hand_palm_pos'] = physics.data.site_xpos[1].copy()
        return obs
    
    @property
    def step_info(self):
        """Compatability function to pipe extra step information for gym compat """
        return self._info

    @property
    def step_count(self):
        return self._step_count

    def get_reward(self, physics):
        reward = 0
        for reward_fn, lambda_r in zip(self._reward_fns, self._reward_wgts):
            r_i, info_i = reward_fn(physics, self._step_count)
            reward += lambda_r * r_i
            self._info.update(info_i)
        return reward

    def get_termination(self, physics):
        for reward_fn in self._reward_fns:
            if reward_fn.check_termination(physics):
                return 0.0
        return None


class SingleObjectTask(Task):
    def __init__(self, object_name, reward_fns, reward_weights=None, random=None):
        self._object_name = object_name
        super().__init__(reward_fns, reward_weights=reward_weights, random=random)
    
    def get_observation(self, physics):
        obs = super().get_observation(physics)
        base_pos = obs['qpos']
        base_vel = obs['qvel']

        hand_poses = physics.body_poses
        hand_com = hand_poses.pos.reshape((-1, 3))
        hand_rot = hand_poses.rot.reshape((-1, 4))
        hand_lv = hand_poses.linear_vel.reshape((-1, 3))
        hand_av = hand_poses.angular_vel.reshape((-1, 3))
        hand_vel = np.concatenate((hand_lv, hand_av), 1)

        object_name = self.object_name
        obj_com = physics.named.data.xipos[object_name].copy()
        obj_rot = physics.named.data.xquat[object_name].copy()
        obj_vel = physics.data.object_velocity(object_name, 'body')
        obj_vel = obj_vel.reshape((1, 6))
        
        full_com = np.concatenate((hand_com, obj_com.reshape((1,3))), 0)
        full_rot = np.concatenate((hand_rot, obj_rot.reshape((1,4))), 0)
        full_vel = np.concatenate((hand_vel, obj_vel), 0)

        obs['position'] = np.concatenate((base_pos[:42], full_com.reshape(-1), full_rot.reshape(-1))).astype(np.float32)
        obs['velocity'] = np.concatenate((base_vel[:42], full_vel.reshape(-1))).astype(np.float32)
        obs['hand_position'] = np.concatenate((base_pos[:30], hand_com.reshape(-1), hand_rot.reshape(-1))).astype(np.float32)
        obs['hand_velocity'] = np.concatenate((base_vel[:30], hand_vel.reshape(-1))).astype(np.float32)
        obs['state'] = np.concatenate((obs['position'], obs['velocity']))
        return obs
    
    @property
    def object_name(self):
        return self._object_name

class ReferenceMotionTask(SingleObjectTask):
    def __init__(self, reference_motion, reward_fns, init_key, rot_aug, random_episode, action, reward_weights=None, random=None):
        self.reference_motion = reference_motion
        self._init_key = init_key
        self._rot_aug = rot_aug
        self._random_episode = random_episode
        self._action = action
        self.goal_obs = False
        self.key_step = self.reference_motion.pregrasp_step
        object_name = reference_motion.object_name
        super().__init__(object_name, reward_fns, reward_weights, random)

    def initialize_episode(self, physics, stage, mode):
        body_names = ["palm", "thdistal_point", "ffdistal_point", "mfdistal_point", "rfdistal_point", "lfdistal_point"]
        for idx in range(len(body_names)):
            body_names[idx] = "adroit/" + body_names[idx]
        self.body_names = body_names
        self.stage = stage
        self.mode = mode
        randomness_scale = 0.25

        if self._random_episode:
            if self._action == "pour":
                final_goal = np.array([-0.025, -0.1, 0.12], dtype=np.float32)
                if self.stage == 0:
                    traj_x = 0.22
                    traj_y = 0.0
                    traj_rot = 0.0
                elif self.stage == 1:
                    traj_x = 0.22 + self._random.uniform(low=-0.06, high=0.04)
                    traj_y = self._random.uniform(low=0.0, high=0.1)
                    traj_rot = 0.0
                elif self.stage == 2:
                    traj_x = 0.22 + self._random.uniform(low=-0.06, high=0.04)
                    traj_y = self._random.uniform(low=0.0, high=0.1)
                    traj_rot = 0.0
            elif self._action == "place":
                final_goal = np.array([0.0, -0.02, 0.12], dtype=np.float32)
                if self.stage == 0:
                    traj_x = 0.22
                    traj_y = 0.0
                    traj_rot = 0.0
                elif self.stage == 1:
                    traj_x = 0.22 + self._random.uniform(low=-0.05, high=0.05)
                    traj_y = self._random.uniform(low=0.0, high=0.1)
                    traj_rot = 0.0
                elif self.stage == 2:
                    traj_x = 0.22 + self._random.uniform(low=-0.05, high=0.05)
                    traj_y = self._random.uniform(low=0.0, high=0.1)
                    traj_rot = 0.0
            else:
                final_goal = np.array([self._random.uniform(low=-0.3, high=0.1), self._random.uniform(low=-0.15, high=0.15), self._random.uniform(low=0.15, high=0.25)], dtype=np.float32)
                if self.stage == 0:
                    traj_x = 0.15 * randomness_scale - 0.1
                    traj_y = 0.0
                    traj_rot = 0.0
                elif self.stage == 1:
                    traj_x = 0.15 * randomness_scale - 0.1
                    traj_y = 0.0
                    traj_rot = self._random.uniform(low=-1/12, high=1/12) * np.pi
                elif self.stage == 2:
                    traj_x = 0.15 * randomness_scale - 0.1
                    traj_y = self._random.uniform(low=-0.15, high=0.15) * randomness_scale
                    traj_rot = self._random.uniform(low=-1/12, high=1/12) * np.pi
        else:
            final_goal = np.zeros(3, dtype=np.float32)
            traj_x = 0.0
            traj_y = 0.0
            traj_rot = 0.0

        traj_trans = np.array([traj_x, traj_y])
        self.reference_motion.reset(traj_rot, traj_trans, final_goal, self._rot_aug, self._random_episode, self._action)
        self.retarget_robot_qpos = self.reference_motion.cur_reference_motion['robot_qpos'].copy()
        self.retarget_robot_jpos = self.reference_motion.cur_reference_motion['robot_pregrasp_jpos'].copy()

        if self._init_key == "pregrasp":
            start_state = np.zeros(36)
            start_state[:30] = self.retarget_robot_qpos[-1]
            self.pregrasp_success = True
            with physics.reset_context():
                physics.model.body_pos[45] = self.reference_motion.cur_reference_motion['object_translation'][0].copy()
                physics.model.body_quat[45] = self.reference_motion.cur_reference_motion['object_orientation'][0].copy()
                physics.model.body_pos[46] = self.reference_motion.cur_reference_motion['object_translation'][-1].copy()
                physics.model.body_quat[46] = self.reference_motion.cur_reference_motion['object_orientation'][-1].copy()
                physics.data.qpos[:] = start_state
                physics.data.qvel[:] = np.zeros(36)
        else:
            self.pregrasp_success = False
            with physics.reset_context():
                physics.model.body_pos[45] = self.reference_motion.cur_reference_motion['object_translation'][0].copy()
                physics.model.body_quat[45] = self.reference_motion.cur_reference_motion['object_orientation'][0].copy()
                physics.model.body_pos[46] = self.reference_motion.final_goal
                physics.model.body_quat[46] = self.reference_motion.cur_reference_motion['object_orientation'][-1].copy()
                if self._action == "pour":
                    physics.model.body_pos[50] = self.reference_motion.cur_reference_motion['object_translation'][0].copy()
                    physics.model.body_quat[50] = self.reference_motion.cur_reference_motion['object_orientation'][0].copy()
                    physics.data.qpos[:] = np.zeros(78)
                    physics.data.qvel[:] = np.zeros(78)
                elif self._action == "place":
                    physics.data.qpos[:] = np.zeros(42)
                    physics.data.qvel[:] = np.zeros(42)
                else:
                    physics.data.qpos[:] = np.zeros(36)
                    physics.data.qvel[:] = np.zeros(36)
        
        return super().initialize_episode(physics)

    def before_step(self, action, physics):
        super().before_step(action, physics)

        if self.get_imitate_state():
            self.goal_obs = True

        if self.get_pregrasp_state() and not self.get_imitate_state():
            self.reference_motion.step()

    def get_termination(self, physics):
        if self.reference_motion.next_done and not self.get_imitate_state():
            return 0.0
        return super().get_termination(physics)

    @property
    def substeps(self):
        return self.reference_motion.substeps

    def get_observation(self, physics):
        obs = super().get_observation(physics)
        obs['goal'] = self.reference_motion.goals.astype(np.float32)
        obs['final_goal'] = self.reference_motion.final_goal.astype(np.float32)
        if self._random_episode:
            obs['state'] = np.concatenate((obs['state'], obs['goal'], obs['hand_palm_pos'] - obs['obj_pos'], (obs['hand_pos'][1:] - obs['obj_pos']).reshape(-1), obs['hand_palm_pos'] - obs['final_goal'], obs['obj_pos'] - obs['final_goal']))
        else:
            obs['state'] = np.concatenate((obs['state'], obs['goal']))
        return obs

    def get_pregrasp_state(self):
        for reward_fn in self._reward_fns:
            return reward_fn.pregrasp_success

    def get_imitate_state(self):
        for reward_fn in self._reward_fns:
            return reward_fn.imitate_success


class ObjMimicTask(ReferenceMotionTask):
    def __init__(self, object_name, data_path, robot_geom_names, object_geom_names, reward_kwargs, append_time, pregrasp_init_key, rot_aug, random_episode, action):
        reference_motion = HandObjectReferenceMotion(object_name, data_path)
        self.robot_geom_names = robot_geom_names
        self.object_geom_names = object_geom_names
        reward_fn = ObjectMimic(**reward_kwargs)
        self._append_time = append_time
        super().__init__(reference_motion, [reward_fn], pregrasp_init_key, rot_aug, random_episode, action)
    
    def get_observation(self, physics):
        obs = super().get_observation(physics)
        
        # append time to observation if needed
        if self._append_time:
            t = self.reference_motion.time
            t = np.array([1, 4, 6, 8]) * t
            t = np.concatenate((np.sin(t), np.cos(t)))
            obs['state'] = np.concatenate((obs['state'], t))
        return obs
