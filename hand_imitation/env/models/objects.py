# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from hand_imitation.env.models import asset_abspath
from .base import MjModel
from dm_control import mjcf
import numpy as np


class ObjectModel(MjModel):
    def __init__(self, base_pos, base_quat, mjcf_model):
        if isinstance(mjcf_model, str):
            mjcf_model = mjcf.from_path(mjcf_model)
    
        if base_pos is not None:
            self._base_pos = np.array(base_pos).copy()
            mjcf_model.worldbody.all_children()[0].pos = self._base_pos
        else:
            self._base_pos = mjcf_model.worldbody.all_children()[0].pos
        
        if base_quat is not None:
            self._base_quat = np.array(base_quat).copy()
            mjcf_model.worldbody.all_children()[0].quat = self._base_quat
        else:
            self._base_quat = mjcf_model.worldbody.all_children()[0].quat
    
        super().__init__(mjcf_model)

    @property
    def start_pos(self):
        return self._base_pos.copy().astype(np.float32)

    @property
    def start_ori(self):
        return self._base_quat.copy().astype(np.float32)


class DAPGHammerObject(ObjectModel):
    def __init__(self, pos=[0, -0.2, 0.035], quat=[0.707388, 0.706825, 0, 0]):
        xml_path = asset_abspath('objects/dapg_hammer.xml')
        super().__init__(pos, quat, xml_path)


def object_generator(path):
    class __XMLObj__(ObjectModel):
        def __init__(self, pos=None, quat=None):
            xml_path = asset_abspath(path)
            super().__init__(pos, quat, xml_path)
    return __XMLObj__


def get_object(name, object_category):
    return object_generator(f"objects/objects/{object_category}/{name}.xml")
