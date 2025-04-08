# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import yaml, cv2
import numpy as np
from dm_control import mjcf
from hand_imitation.env.models import asset_abspath
from .motion_util import PoseAndVelocity


ROBOT_CONFIGS = yaml.safe_load(open(asset_abspath('robot_config.yaml'), 'r'))


def physics_from_mjcf(env, robot_name=None):
    mjcf_model = env.mjcf_model
    if robot_name is None:
        xml = mjcf_model.to_xml_string()
        if 'adroit' in xml:
            robot_name = 'adroit'
        else:
            raise NotImplementedError
    
    if robot_name == 'adroit':
        return AdroitPhysics.from_mjcf_model(mjcf_model)
    raise NotImplementedError


class _Physics(mjcf.Physics):
    _ROBOT_NAME = None

    @property
    def body_poses(self):
        pos, rot, lvel, avel = [], [], [], []
        for b in ROBOT_CONFIGS[self._ROBOT_NAME]['BODIES']:
            b = '{}/{}'.format(self._ROBOT_NAME, b)
            pos.append(self.named.data.geom_xpos[b][None])
            rot.append(self.named.data.xquat[b][None])
            lv, av = self.data.object_velocity(b, 'geom')
            lvel.append(lv[None])
            avel.append(av[None])
        pos, rot, lvel, avel = [np.concatenate(ar, 0) for ar in (pos, rot, lvel, avel)]
        return PoseAndVelocity(pos, rot, lvel, avel)

    @property
    def fingertip_indices(self):
        return list(ROBOT_CONFIGS[self._ROBOT_NAME]['FINGERTIP_INDICES'])
    
    @property
    def n_hand_contacts(self):
        num = 0
        for c in self.data.contact:
            name1 = self.model.id2name(c.geom1, 'geom')
            name2 = self.model.id2name(c.geom2, 'geom')

            contact1 = self._ROBOT_NAME in name1 and '_contact' in name2
            contact2 = self._ROBOT_NAME in name2 and '_contact' in name1
            contact3 = self._ROBOT_NAME in name1 and self._ROBOT_NAME in name2
            if contact1 or contact2 or contact3:
                num += 1
        return num

    def set_control(self, action):
        super().set_control(action)
        # applies gravity compensation to robot joints
        grav_comp = self.named.data.qfrc_bias[:self.adim]
        self.named.data.qfrc_applied[:self.adim] = grav_comp

    @property
    def adim(self):
        return self.model.nu
    
    @property
    def ctrl_range(self):
        return self.model.actuator_ctrlrange


class AdroitPhysics(_Physics):
    _ROBOT_NAME = 'adroit'
