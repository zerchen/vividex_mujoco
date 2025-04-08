import os

# logic for retrieving assets
def asset_abspath(resource_path):
    envs_path = os.path.dirname(__file__)
    asset_folder_path = os.path.join(envs_path, 'assets')
    return os.path.join(asset_folder_path, resource_path)

from .arena import TableEnv, EmptyEnv
from .robots import get_robot
from .objects import get_object
from .physics import physics_from_mjcf, ROBOT_CONFIGS
from .control import Environment, ObjMimicTask
from .reference import HandObjectReferenceMotion, HandReferenceMotion
