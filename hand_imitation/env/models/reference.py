# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from .motion_util import Pose, PoseAndVelocity


class HandReferenceMotion(object):
    def __init__(self, motion_file, start_step=0):
        self._load_motion(motion_file)
        self._substeps = int(self._reference_motion['SIM_SUBSTEPS'])                
        self._data_substeps = self._reference_motion.get('DATA_SUBSTEPS', self._substeps)              
        self._step, self._start_step = 0, int(start_step)
        self.pregrasp_step = self._reference_motion['pregrasp_step']

        self._reference_motion['robot_qpos'] = self._reference_motion['robot_qpos'][:self.pregrasp_step + 1]
        self._reference_motion['robot_pregrasp_jpos'] = self._reference_motion['robot_jpos'][:self.pregrasp_step + 1].copy()
        self._reference_motion['robot_jpos'] = self._reference_motion['robot_jpos'][self.pregrasp_step:]
        self._reference_motion['object_translation'] = self._reference_motion['object_translation'][self.pregrasp_step:]
        self._reference_motion['object_orientation'] = self._reference_motion['object_orientation'][self.pregrasp_step:]
        self._reference_motion['length'] = self._reference_motion['length'] - self.pregrasp_step

        # first canonicalize the traj
        init_object_translation = self._reference_motion['object_translation'][0].copy()
        init_object_orientation = self._reference_motion['object_orientation'][0].copy()
        self._reference_motion['object_translation'][:, :2] -= init_object_translation[:2]
        self._reference_motion['robot_jpos'][:, :, :2] -= init_object_translation[:2]
        self._reference_motion['robot_pregrasp_jpos'][:, :, :2] -= init_object_translation[:2]
        self._reference_motion['robot_qpos'][:, 0] += init_object_translation[0]
        self._reference_motion['robot_qpos'][:, 2] -= init_object_translation[1]

        self.cur_reference_motion = copy.deepcopy(self._reference_motion)
        self._final_goal = np.zeros(3, dtype=np.float32)
    
    def _load_motion(self, motion_file):
        motion_file = np.load(motion_file, allow_pickle=True)
        self._reference_motion = {k:v for k, v in motion_file.items()}

    def reset(self, init_rot, init_trans, final_goal, rot_aug, random_episode, action):
        self._step = 0
        self._final_goal = final_goal
        self.cur_reference_motion = copy.deepcopy(self._reference_motion)

        # then rotate the traj
        if rot_aug:
            rot_matrix = np.array([[np.cos(init_rot), -np.sin(init_rot), 0], [np.sin(init_rot), np.cos(init_rot), 0], [0, 0, 1]])
            self.cur_reference_motion['object_translation'] = (rot_matrix @ self.cur_reference_motion['object_translation'].transpose(1, 0)).transpose(1, 0)
            for idx in range(self.cur_reference_motion['object_orientation'].shape[0]):
                self.cur_reference_motion['object_orientation'][idx] = Quaternion(matrix=(rot_matrix @ Quaternion(self.cur_reference_motion['object_orientation'][idx]).rotation_matrix)).elements
            self.cur_reference_motion['robot_jpos'] = (rot_matrix[None] @ self.cur_reference_motion['robot_jpos'].transpose(0, 2, 1)).transpose(0, 2, 1)
            self.cur_reference_motion['robot_pregrasp_jpos'] = (rot_matrix[None] @ self.cur_reference_motion['robot_pregrasp_jpos'].transpose(0, 2, 1)).transpose(0, 2, 1)
            ori_pos = np.zeros_like(self.cur_reference_motion['robot_qpos'][:, :2])
            ori_pos[:, 0] = -self.cur_reference_motion['robot_qpos'][:, 0]
            ori_pos[:, 1] = self.cur_reference_motion['robot_qpos'][:, 1] - 0.7
            new_pos = (rot_matrix[:2, :2] @ ori_pos.transpose(1, 0)).transpose(1, 0)
            self.cur_reference_motion['robot_qpos'][:, 0] -= (new_pos[:, 0] - ori_pos[:, 0])
            self.cur_reference_motion['robot_qpos'][:, 2] += (new_pos[:, 1] - ori_pos[:, 1])
            rot_robot = R.from_euler('XYZ', self.cur_reference_motion['robot_qpos'][:, [3, 4, 5]]).as_euler('YXZ')
            rot_robot[:, 0] += init_rot
            self.cur_reference_motion['robot_qpos'][:, [3, 4, 5]] = R.from_euler('YXZ', rot_robot).as_euler('XYZ')

        # then translate the traj
        self.cur_reference_motion['object_translation'][:, :2] += init_trans
        self.cur_reference_motion['robot_jpos'][:, :, :2] += init_trans
        self.cur_reference_motion['robot_pregrasp_jpos'][:, :, :2] += init_trans
        self.cur_reference_motion['robot_qpos'][:, 0] -= init_trans[0]
        self.cur_reference_motion['robot_qpos'][:, 2] += init_trans[1]

        if not random_episode:
            self._final_goal = self.cur_reference_motion['object_translation'][-1]
        elif action == "pour":
            keyframe = np.where((self.cur_reference_motion['object_translation'][:, 2] - self.cur_reference_motion['object_translation'][0, 2]) > 0.12)[0][0] + 1
            self.cur_reference_motion['object_translation'] = self.cur_reference_motion['object_translation'][:keyframe]
            self.cur_reference_motion['object_orientation'] = self.cur_reference_motion['object_orientation'][:keyframe]
            self.cur_reference_motion['robot_jpos'] = self.cur_reference_motion['robot_jpos'][:keyframe]
            self._final_goal[2] = self.cur_reference_motion['object_translation'][-1, 2]
            final_rot = (R.from_rotvec(-2 * np.pi / 3 * np.array([0, 1, 0])) * R.from_quat(self.cur_reference_motion['object_orientation'][-1][[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]

            dist_vec = self._final_goal - self.cur_reference_motion['object_translation'][-1]
            unit_dist_vec = dist_vec / np.linalg.norm(dist_vec)
            relocate_step = 0.02
            num_step = int(np.linalg.norm(dist_vec) // relocate_step)
            init_rot = R.from_quat(self.cur_reference_motion['object_orientation'][-1][[1, 2, 3, 0]])
            rotation_step = -2 * np.pi / 3 / num_step

            syn_object_translation = []
            syn_object_orientation = []
            syn_robot_jpos = []
            for idx in range(num_step):
                step_size = (idx + 1) * relocate_step
                syn_object_translation.append(self.cur_reference_motion['object_translation'][-1] + step_size * unit_dist_vec)
                rot_size = (idx + 1) * rotation_step
                syn_object_orientation.append((R.from_rotvec(rot_size * np.array([0, 1, 0])) * init_rot).as_quat()[[3, 0, 1, 2]])
                cur_joint = self.cur_reference_motion['robot_jpos'][-1] - self.cur_reference_motion['object_translation'][-1]
                rotmat = R.from_rotvec(rot_size * np.array([0, 1, 0])).as_matrix()
                cur_joint = (rotmat @ cur_joint.transpose(1, 0)).transpose(1, 0) + syn_object_translation[-1]
                syn_robot_jpos.append(cur_joint)
            
            rot_size = -2 * np.pi / 3
            syn_object_translation.append(self._final_goal)
            syn_object_orientation.append(final_rot)
            cur_joint = self.cur_reference_motion['robot_jpos'][-1] - self.cur_reference_motion['object_translation'][-1]
            rotmat = R.from_rotvec(rot_size * np.array([0, 1, 0])).as_matrix()
            cur_joint = (rotmat @ cur_joint.transpose(1, 0)).transpose(1, 0) + syn_object_translation[-1]
            syn_robot_jpos.append(cur_joint)
            
            self.cur_reference_motion['object_translation'] = np.concatenate((self.cur_reference_motion['object_translation'], np.array(syn_object_translation)))
            self.cur_reference_motion['object_orientation'] = np.concatenate((self.cur_reference_motion['object_orientation'], np.array(syn_object_orientation)))
            self.cur_reference_motion['robot_jpos'] = np.concatenate((self.cur_reference_motion['robot_jpos'], np.array(syn_robot_jpos)))
            self.cur_reference_motion['length'] = len(self.cur_reference_motion['object_translation'])
        elif action == "place":
            keyframe = np.where((self.cur_reference_motion['object_translation'][:, 2] - self.cur_reference_motion['object_translation'][0, 2]) > 0.24)[0][0] + 1
            self.cur_reference_motion['object_translation'] = self.cur_reference_motion['object_translation'][:keyframe]
            self.cur_reference_motion['object_orientation'] = self.cur_reference_motion['object_orientation'][:keyframe]
            self.cur_reference_motion['robot_jpos'] = self.cur_reference_motion['robot_jpos'][:keyframe]
            self._final_goal[2] = self.cur_reference_motion['object_translation'][-1, 2]
            final_rot = (R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0])) * R.from_quat(self.cur_reference_motion['object_orientation'][-1][[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]

            dist_vec = self._final_goal - self.cur_reference_motion['object_translation'][-1]
            unit_dist_vec = dist_vec / np.linalg.norm(dist_vec)
            relocate_step = 0.02
            num_step = int(np.linalg.norm(dist_vec) // relocate_step)
            init_rot = R.from_quat(self.cur_reference_motion['object_orientation'][-1][[1, 2, 3, 0]])
            rotation_step = -np.pi / 2 / num_step

            syn_object_translation = []
            syn_object_orientation = []
            syn_robot_jpos = []
            for idx in range(num_step):
                step_size = (idx + 1) * relocate_step
                syn_object_translation.append(self.cur_reference_motion['object_translation'][-1] + step_size * unit_dist_vec)
                rot_size = (idx + 1) * rotation_step
                syn_object_orientation.append((R.from_rotvec(rot_size * np.array([1, 0, 0])) * init_rot).as_quat()[[3, 0, 1, 2]])
                cur_joint = self.cur_reference_motion['robot_jpos'][-1] - self.cur_reference_motion['object_translation'][-1]
                rotmat = R.from_rotvec(rot_size * np.array([1, 0, 0])).as_matrix()
                cur_joint = (rotmat @ cur_joint.transpose(1, 0)).transpose(1, 0) + syn_object_translation[-1]
                syn_robot_jpos.append(cur_joint)

            descent_vec = np.array([0, 0, -0.02])
            for idx in range(8):
                syn_object_translation.append(syn_object_translation[-1] + descent_vec)
                syn_object_orientation.append(syn_object_orientation[-1])
                syn_robot_jpos.append(syn_robot_jpos[-1] + descent_vec[None, :])
            
            self.cur_reference_motion['object_translation'] = np.concatenate((self.cur_reference_motion['object_translation'], np.array(syn_object_translation)))
            self._final_goal = self.cur_reference_motion['object_translation'][-1]
            self.cur_reference_motion['object_orientation'] = np.concatenate((self.cur_reference_motion['object_orientation'], np.array(syn_object_orientation)))
            self.cur_reference_motion['robot_jpos'] = np.concatenate((self.cur_reference_motion['robot_jpos'], np.array(syn_robot_jpos)))
            self.cur_reference_motion['length'] = len(self.cur_reference_motion['object_translation'])
        else:
            keyframe = np.where((self.cur_reference_motion['object_translation'][:, 2] - self.cur_reference_motion['object_translation'][0, 2]) > 0.1)[0][0] + 1
            self.cur_reference_motion['object_translation'] = self.cur_reference_motion['object_translation'][:keyframe]
            self.cur_reference_motion['object_orientation'] = self.cur_reference_motion['object_orientation'][:keyframe]
            self.cur_reference_motion['robot_jpos'] = self.cur_reference_motion['robot_jpos'][:keyframe]

            dist_vec = self._final_goal - self.cur_reference_motion['object_translation'][-1]
            unit_dist_vec = dist_vec / np.linalg.norm(dist_vec)
            relocate_step = 0.02
            num_step = int(np.linalg.norm(dist_vec) // relocate_step)

            syn_object_translation = []
            syn_object_orientation = []
            syn_robot_jpos = []
            for idx in range(num_step):
                step_size = (idx + 1) * relocate_step
                syn_object_translation.append(self.cur_reference_motion['object_translation'][-1] + step_size * unit_dist_vec)
                syn_object_orientation.append(self.cur_reference_motion['object_orientation'][-1])
                syn_robot_jpos.append(self.cur_reference_motion['robot_jpos'][-1] + step_size * unit_dist_vec)
            
            syn_object_translation.append(self._final_goal)
            syn_object_orientation.append(self.cur_reference_motion['object_orientation'][-1])
            syn_robot_jpos.append(self.cur_reference_motion['robot_jpos'][-1] + dist_vec)
            
            self.cur_reference_motion['object_translation'] = np.concatenate((self.cur_reference_motion['object_translation'], np.array(syn_object_translation)))
            self.cur_reference_motion['object_orientation'] = np.concatenate((self.cur_reference_motion['object_orientation'], np.array(syn_object_orientation)))
            self.cur_reference_motion['robot_jpos'] = np.concatenate((self.cur_reference_motion['robot_jpos'], np.array(syn_robot_jpos)))
            self.cur_reference_motion['length'] = len(self.cur_reference_motion['object_translation'])

    def step(self):
        self._check_valid_step()
        self._step += self._data_substeps
    
    def revert(self):
        self._step -= self._data_substeps
        self._check_valid_step()

    def __len__(self):
        return self.length

    @property
    def t(self):
        return self._step
    
    @property
    def data_substep(self):
        return self._data_substeps

    @property
    def time(self):
        return float(self._step) / self.length

    @property
    def substeps(self):
        return self._substeps

    @property
    def done(self):
        assert self._step is not None, "Motion must be reset before it can be done"
        return self._step >= self.length
    
    @property
    def next_done(self):
        assert self._step is not None, "Motion must be reset before it can be done"
        return self._step >= self.length - self._data_substeps

    @property
    def n_left(self):
        assert self._step is not None, "Motion must be reset before lengths calculated"
        n_left = (self.length - self._step) / float(self._data_substeps) - 1
        return int(max(n_left, 0))
    
    @property
    def n_steps(self):
        n_steps = self.length / float(self._data_substeps) - 1
        return int(max(n_steps, 0))

    @property
    def length(self):
        if 'length' in self.cur_reference_motion:
            return self.cur_reference_motion['length']
        return self.cur_reference_motion['s'].shape[0]
    
    @property
    def start_step(self):
        return self._start_step

    def _check_valid_step(self):
        assert not self.done, "Attempting access data and/or step 'done' motion"
        assert self._step >= self._start_step, "step must be at least start_step"
    
    def __getitem__(self, key):
        value = copy.deepcopy(self.cur_reference_motion[key])
        if not isinstance(value, np.ndarray):
            return value
        if len(value.shape) >= 2:
            return value[self._start_step::self._data_substeps]
        return value


class HandObjectReferenceMotion(HandReferenceMotion):
    def __init__(self, object_name, motion_file):
        super().__init__(motion_file)
        self._object_name = object_name

    @property
    def object_name(self):
        return self._object_name

    @property
    def hand_jnts(self):
        self._check_valid_step()
        return self.cur_reference_motion['robot_jpos'][self._step].copy()

    @property
    def object_pos(self):
        self._check_valid_step()
        return self.cur_reference_motion['object_translation'][self._step].copy()
    
    @property
    def floor_z(self):
        return float(self.cur_reference_motion['object_translation'][0, 2])
    
    @property
    def object_rot(self):
        self._check_valid_step()
        return self.cur_reference_motion['object_orientation'][self._step].copy()
    
    @property
    def object_pose(self):
        pos = self.object_pos[None]
        rot = self.object_rot[None]
        return Pose(pos, rot)

    @property
    def final_goal(self):
        pos = self._final_goal.copy()
        return pos

    @property
    def goals(self):
        g = []
        for i in [1, 5, 10]:
            i = min(self._step + i, self.length-1)
            for k in ('object_orientation', 'object_translation'):
                g.append(self.cur_reference_motion[k][i].flatten())
        return np.concatenate(g)
