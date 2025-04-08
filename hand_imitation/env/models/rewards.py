# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import abc
import copy
import numpy as np
from .motion_util import rotation_distance
from scipy.spatial.transform import Rotation as R
from hand_imitation.env.models.assets.objects.object_stats import OBJECT_HEIGHT


def get_reward(name):
    if name == 'objectmimic':
        return ObjectMimic
    if name == 'dummy':
        return Dummy
    raise ValueError("Reward {} not supported!".format(name)) 


def norm2(x):
    return np.sum(np.square(x))


class _ErrorTracker:
    def __init__(self, targets, thresh=0.01, start=0):
        self._targets = targets.copy()
        self._values = np.zeros_like(self._targets)
        self._thresh, self._i = thresh, 0
        self._start = start
    
    def append(self, v):
        if self._i >= len(self._values):
            return
        self._values[self._i:] = v[None]
        self._i += 1

    def reset(self):
        self._values = np.zeros_like(self._targets)
        self._i = 0
    
    @property
    def N(self):
        return len(self._targets) - self._start

    @property
    def error(self):
        v, t = self._values[self._start:], self._targets[self._start:]
        deltas = np.sqrt(np.sum(np.square(v - t), axis=-1))
        deltas = np.mean(deltas.reshape((self.N, -1)), axis=1)
        return np.sum(deltas) / self.N

    @property
    def success(self):
        v, t = self._values[self._start:], self._targets[self._start:]
        deltas = np.sqrt(np.sum(np.square(v - t), axis=-1))
        if len(deltas.shape) > 1:
            deltas = np.mean(deltas.reshape((self.N, -1)), axis=1)
        return np.mean(deltas <= self._thresh)

    @property
    def success_goal(self):
        v, t = self._values[self._start:], self._targets[self._start:]
        deltas = np.sqrt(np.sum(np.square(v - t), axis=-1))
        if len(deltas.shape) > 1:
            deltas = np.mean(deltas.reshape((self.N, -1)), axis=1)
        return deltas[-1] <= self._thresh * 5


class RewardFunction(abc.ABC):
    def __init__(self, **override_hparams):
        """
        Overrides default hparams with values passed into initializer
        """
        params = copy.deepcopy(self.DEFAULT_HPARAMS)
        for k, v in override_hparams.items():
            assert k in params, "Param {} does not exist in struct".format(k)
            params[k] = v
        
        for k, v in params.items():
            setattr(self, k, v)

    @abc.abstractproperty
    def DEFAULT_HPARAMS(self):
        """
        Returns default hyperparamters for reward function
        """
    
    def initialize_rewards(self, parent_task, physics):
        """
        Gets parent task and sets constants as required from it
        """
        self._parent_task = parent_task

    @abc.abstractmethod
    def get_reward(self, physics, step):
        """
        Calculates reward and success stats from phyiscs data
        Returns reward, info_dict
        """

    @abc.abstractmethod
    def check_termination(self, physics):
        """
        Checks if trajectory should terminate
        Returns terminate_flag
        """
    
    def __call__(self, physics, step):
        return self.get_reward(physics, step)


class Dummy(RewardFunction):
    @property
    def DEFAULT_HPARAMS(self):
        return dict()
    
    def get_reward(self, _):
        return 0.0, dict()
    
    def check_termination(self, _):
        return False


class ObjectMimic(RewardFunction):
    @property
    def DEFAULT_HPARAMS(self):
        return  {
                    'hand_finger_err_thresh': 0.05,
                    'pregrasp_steps': 12,
                    'manipulation_steps': 60,
                    'hand_err_scale': 10,
                    'obj_err_scale': 50,
                    'object_reward_scale': 10,
                    'lift_bonus_thresh': 0.02,
                    'lift_bonus_mag': 2.5,
                    'obj_com_term': 0.25,
                    'n_envs': 1,
                    'obj_reward_ramp': 0,
                    'obj_reward_start': 999
                }

    def check_contacts(self, physics, geoms_1, geoms_2):
        for contact in physics.data.contact[:physics.data.ncon]:
            # check contact geom in geoms
            c1_in_g1 = physics.model.geom(contact.geom1).name in geoms_1
            c2_in_g2 = physics.model.geom(contact.geom2).name in geoms_2

            # check contact geom in geoms (flipped)
            c2_in_g1 = physics.model.geom(contact.geom2).name in geoms_1
            c1_in_g2 = physics.model.geom(contact.geom1).name in geoms_2
            if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
                return True
        return False
    
    def initialize_rewards(self, parent_task, physics):
        self._mode = parent_task.mode
        self._step_count = parent_task.step_count
        self._reference_motion = parent_task.reference_motion
        self._body_names = parent_task.body_names
        self._object_name = self._reference_motion.object_name
        self.floor_z = physics.named.data.xipos[self._object_name][2].copy()
        self._lift_z = self.floor_z + self.lift_bonus_thresh
        self.retarget_robot_qpos = parent_task.retarget_robot_qpos
        self.retarget_robot_jpos = parent_task.retarget_robot_jpos
        self.key_step = parent_task.key_step
        self.start_step = np.where(self.retarget_robot_qpos[:, 2] > 0)[0][0]
        self.pregrasp_steps = self.key_step - self.start_step + 1
        self.pregrasp_success = parent_task.pregrasp_success
        if self.pregrasp_success:
            self.obj_reward_start = 0
        self.imitate_success = False
        self.robot_geom_names = parent_task.robot_geom_names
        self.object_geom_names = parent_task.object_geom_names
        self.stage = parent_task.stage
        self.relocate = parent_task._random_episode
        self.imitate_steps = self.pregrasp_steps + self._reference_motion.length - 1
        if parent_task._action  == "pour":
            self.manipulation_steps = 100
        elif parent_task._action  == "place":
            self.manipulation_steps = 80
        else:
            self.manipulation_steps = 60
        self.relocate_success_steps = self.manipulation_steps

        # register metric tracking data
        self._obj = _ErrorTracker(self._reference_motion.cur_reference_motion['object_translation'][1:])
        self._hand = _ErrorTracker(self._reference_motion.cur_reference_motion['robot_jpos'][1:, [1, 2, 3, 4, 5]], thresh=0.05)

    def get_reward(self, physics, step):
        self._step_count = step
        object_table_contact = self.check_contacts(physics, self.object_geom_names, ['table_contact'])
        first_object_contact = self.check_contacts(physics, self.robot_geom_names[6], self.object_geom_names)
        middle_object_contact = self.check_contacts(physics, self.robot_geom_names[9], self.object_geom_names)
        ring_object_contact = self.check_contacts(physics, self.robot_geom_names[12], self.object_geom_names)
        little_object_contact = self.check_contacts(physics, self.robot_geom_names[16], self.object_geom_names)
        thumb_object_contact = self.check_contacts(physics, self.robot_geom_names[19], self.object_geom_names)
        is_contact = self.check_contacts(physics, self.robot_geom_names, self.object_geom_names)

        # get final targets
        tgt_obj_pos = self._reference_motion.final_goal

        # get real values from physics object
        robot_qpos = physics.data.qpos[:30].copy()
        obj_com = physics.named.data.xipos[self._object_name].copy()
        obj_rot = physics.named.data.xquat[self._object_name].copy()

        is_lift = (obj_com[2] >= self.floor_z + 0.09) and (not object_table_contact)

        # get real values from adroit hand
        hand_jnts = np.zeros((len(self._body_names), 3), dtype=np.float32)
        for idx in range(len(self._body_names)):
            hand_jnts[idx] = physics.named.data.xpos[self._body_names[idx]].copy()

        # get targets from reference object
        tgt_obj_com = self._reference_motion.object_pos
        tgt_obj_rot = self._reference_motion.object_rot

        # calculate both object "matching" reward
        obj_com_err = np.sqrt(norm2(tgt_obj_com - obj_com))
        obj_rot_err = rotation_distance(obj_rot, tgt_obj_rot) / np.pi

        # get targets from hand motions
        tgt_hand_jnts = self._reference_motion.hand_jnts

        # calculate hand joints "matching" reward
        hand_jnt_err = np.mean(np.linalg.norm(hand_jnts[[1, 2, 3, 4, 5]] - tgt_hand_jnts[[1, 2, 3, 4, 5]], axis=1))

        if self._step_count == 1:
            self._obj.append(obj_com)
            self._hand.append(hand_jnts[[1, 2, 3, 4, 5]])
            info = {
                'stage': self.stage,
                'time_frac': self._reference_motion.time,
                'pregrasp_steps': self.pregrasp_steps,
                'pregrasp_success': 1.0 if self.pregrasp_success else 0.0,
                'imitate_obj_err': self._obj.error,
                'imitate_obj_success': self._obj.success,
                'imitate_goal_success': self._obj.success_goal,
                'imitate_step_obj_err': obj_com_err,
                'imitate_hand_err': self._hand.error,
                'imitate_hand_success': self._hand.success,
                'imitate_step_hand_err': hand_jnt_err
            }
            if self.relocate:
                info['relocate_step_obj_err'] = np.linalg.norm(obj_com - tgt_obj_pos)
                info['relocate_step_hand_err'] = np.linalg.norm(hand_jnts[0] - tgt_obj_pos)
                info['imitate_steps'] = self.imitate_steps
            self._obj.reset()
            self._hand.reset()

        if not self.pregrasp_success:
            if self._step_count <= self.pregrasp_steps:
                hand_jpos_err = np.mean(np.linalg.norm(hand_jnts[[0, 1, 2, 3, 4, 5]] - self.retarget_robot_jpos[self.start_step + self._step_count - 1][[0, 1, 2, 3, 4, 5]], axis=1))
            reward = 10 * np.exp(-self.hand_err_scale * hand_jpos_err)

            if self._step_count > 1:
                info = dict()
            info['pregrasp_jpos_err'] = hand_jpos_err
        else:
            if not self.imitate_success:
                self._obj.append(obj_com)
                self._hand.append(hand_jnts[[1, 2, 3, 4, 5]])

                reward = 0.0
                if is_contact:
                    reward += np.sum([thumb_object_contact, first_object_contact, middle_object_contact, ring_object_contact, little_object_contact]) * 0.5

                    obj_reward = np.exp(-self.obj_err_scale * (obj_com_err + 0.1 * obj_rot_err))
                    lift_bonus = (tgt_obj_com[2] >= self._lift_z) and (obj_com[2] >= self._lift_z) and (not object_table_contact)
                    hand_bonus = norm2(obj_com - tgt_obj_com) < 0.01 ** 2
                    hand_reward = 4.0 * np.exp(-self.hand_err_scale * hand_jnt_err)

                    obj_scale = self._object_reward_scale()
                    reward = reward + obj_scale * obj_reward + self.lift_bonus_mag * float(lift_bonus) + hand_reward * float(hand_bonus)

                if self._step_count > 1:
                    info = dict()
                info['time_frac'] = self._reference_motion.time
                info['pregrasp_success'] = 1.0
                info['pregrasp_steps'] = self.obj_reward_start
                info['imitate_obj_err'] = self._obj.error
                info['imitate_obj_success'] = self._obj.success
                info['imitate_goal_success'] = self._obj.success_goal
                info['imitate_step_obj_err'] = obj_com_err
                info['imitate_hand_err'] = self._hand.error
                info['imitate_hand_success'] = self._hand.success
                info['imitate_step_hand_err'] = hand_jnt_err
            else:
                reward = 0.0
                if is_contact:
                    reward += np.sum([thumb_object_contact, first_object_contact, middle_object_contact, ring_object_contact, little_object_contact]) * 0.5

                    obj_reward = np.exp(-self.obj_err_scale * (obj_com_err + 0.1 * obj_rot_err))
                    lift_bonus = (tgt_obj_com[2] >= self._lift_z) and (obj_com[2] >= self._lift_z) and (not object_table_contact)
                    hand_bonus = norm2(obj_com - tgt_obj_com) < 0.01 ** 2
                    hand_reward = 4.0 * np.exp(-self.hand_err_scale * hand_jnt_err)

                    obj_scale = self._object_reward_scale()
                    reward = reward + obj_scale * obj_reward + self.lift_bonus_mag * float(lift_bonus) + hand_reward * float(hand_bonus)

                if self._step_count > 1:
                    info = dict()
                info['relocate_step_obj_err'] = obj_com_err
                info['relocate_step_hand_err'] = hand_jnt_err

        if not self.imitate_success and self.pregrasp_success and self.relocate and self._step_count == self.pregrasp_steps + self._reference_motion.length - 1:
            if (is_contact and self._obj.success_goal) or self._mode == "test":
                self.imitate_success = True

        if not self.pregrasp_success and self._step_count == self.pregrasp_steps:
            if (hand_jpos_err < self.hand_finger_err_thresh) or self._mode == "test":
                self.pregrasp_success = True
                self.obj_reward_start = self._step_count

        return reward, info

    def check_termination(self, physics):
        if self._mode == "test":
            return self._step_count >= self.manipulation_steps
        else:
            if self.relocate:
                if self.pregrasp_success:
                    if self.imitate_success:
                        is_contact = self.check_contacts(physics, self.robot_geom_names, self.object_geom_names)
                        return (self._step_count >= self.manipulation_steps) or (not is_contact)
                    else:
                        # terminate if object delta greater than threshold
                        obj_com = physics.named.data.xipos[self._object_name].copy()
                        tgt_obj_com = self._reference_motion.object_pos
                        return (norm2(obj_com - tgt_obj_com) >= self.obj_com_term ** 2) or (self._step_count >= self.imitate_steps)
                else:
                    is_contact = self.check_contacts(physics, self.robot_geom_names, self.object_geom_names)
                    return self._step_count >= self.pregrasp_steps or is_contact
            else:
                if self.pregrasp_success:
                    # terminate if object delta greater than threshold
                    obj_com = physics.named.data.xipos[self._object_name].copy()
                    tgt_obj_com = self._reference_motion.object_pos
                    return norm2(obj_com - tgt_obj_com) >= self.obj_com_term ** 2
                else:
                    is_contact = self.check_contacts(physics, self.robot_geom_names, self.object_geom_names)
                    return self._step_count >= self.pregrasp_steps or is_contact

    def _object_reward_scale(self):
        if self.obj_reward_ramp > 0:
            delta = self._step_count * self.n_envs - self.obj_reward_start
            delta /= float(self.obj_reward_ramp)
        else:
            delta = 1.0 if self._step_count >= self.obj_reward_start else 0.0
        return self.object_reward_scale * min(max(delta, 0), 1)
