import numpy as np
from dm_control import viewer
from hand_imitation.env.models import Environment, ObjMimicTask
from hand_imitation.env.models import TableEnv, asset_abspath, get_robot, get_object, physics_from_mjcf


def create_env(name, robot_name="adroit", task_kwargs={}, environment_kwargs={}):
    sim_env = TableEnv()
    object_category = name.split('-')[0]

    robot_model = get_robot(robot_name)(limp=False)
    robot_mesh_list = robot_model.mjcf_model.find_all('geom')
    robot_geom_names = [geom.get_attributes()['name'] for geom in robot_mesh_list]
    robot_geom_names = [f'{robot_name}/{name}' for name in robot_geom_names if 'C' in name]
    sim_env.attach(robot_model)

    data_path = asset_abspath(f"objects/trajectories/{object_category}/{name}.npz")
    traj_data = np.load(data_path)
    object_name = traj_data["object_name"]
    init_obj_pos = traj_data["object_translation"][0].tolist()
    init_obj_quat = traj_data["object_orientation"][0].tolist()

    if object_category == "ycb":
        object_model = get_object(object_name, object_category)(pos=init_obj_pos, quat=init_obj_quat)
    elif object_category == "hoi4d":
        object_model = get_object(object_name, "ycb")(pos=init_obj_pos, quat=init_obj_quat)
    object_model.mjcf_model.worldbody.add('body', name='object_marker', pos=np.array([0.2, 0.2, 0.2]))
    object_model.mjcf_model.worldbody.body['object_marker'].add('geom', contype='0', conaffinity='0', mass='0', name='target_visual', mesh=object_model.mjcf_model.worldbody.body['object_entity'].geom['entity_visual'].mesh, rgba=np.array([0, 1, 0, 0.125]))
    object_model.mjcf_model.worldbody.body['object_marker'].geom['target_visual'].type = "mesh"

    object_mesh_list = object_model.mjcf_model.find_all('geom')
    object_geom_names = [geom.get_attributes()['name'] for geom in object_mesh_list]
    object_geom_names = [f'{object_name}/{name}' for name in object_geom_names if 'contact' in name]
    object_name = '{}/object_entity'.format(object_model.mjcf_model.model)
    sim_env.attach(object_model)

    if task_kwargs['action'] == "pour":
        tank_model = get_object("water_tank", "common")(pos=np.array([-0.08, -0.1, 0.03]), quat=np.array([1, 0, 0, 0]))
        sim_env.attach(tank_model)
        water_model = get_object("water", "common")(pos=init_obj_pos, quat=init_obj_quat)
        sim_env.attach(water_model)

    if task_kwargs['action'] == "place":
        mug_model = get_object("big_mug", "common")(pos=np.array([0.01, 0.0, 0.06]), quat=np.array([1, 0, 0, 0]))
        sim_env.attach(mug_model)

    physics = physics_from_mjcf(sim_env)

    task = ObjMimicTask(object_name, data_path, robot_geom_names, object_geom_names, task_kwargs['reward_kwargs'], task_kwargs['append_time'], task_kwargs['pregrasp'], task_kwargs['rot_aug'], task_kwargs['random_episode'], task_kwargs['action'])
    env = Environment(physics, task, n_sub_steps=task.substeps, **environment_kwargs)

    return env

def create_model(robot_name="adroit"):
    sim_env = TableEnv()

    robot_model = get_robot(robot_name)(limp=False)
    sim_env.attach(robot_model)
    physics = physics_from_mjcf(sim_env)

    return physics
