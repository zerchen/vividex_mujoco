import numpy as np
import os
import trimesh
from tqdm import tqdm
from mesh_to_sdf import sample_sdf_near_surface


def compute_unit_sphere_transform(mesh):
    # the transformation applied by mesh_to_sdf.scale_to_unit_sphere(mesh)
    translation = -mesh.bounding_box.centroid
    scale = 1 / np.max(np.linalg.norm(mesh.vertices + translation, axis=1))
    return translation, scale

data_root = "/gpfswork/rech/ets/uqm11hr/code/dex_sim/hand_imitation/env/models/assets/ycb/visual"
output_dir = "/gpfswork/rech/ets/uqm11hr/code/dex_sim/demonstrations/ycb_rest_pc"
output_sdf_dir = "/gpfswork/rech/ets/uqm11hr/code/dex_sim/demonstrations/ycb_sdf"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_sdf_dir, exist_ok=True)

for foldername in tqdm(os.listdir(data_root)):
    obj_mesh = trimesh.load(os.path.join(data_root, foldername, 'textured_simple.stl'))
    obj_points, _ = trimesh.sample.sample_surface(obj_mesh, 1000)
    obj_name = '_'.join(foldername.split('_')[1:])
    np.save(os.path.join(output_dir, obj_name + ".npy"), obj_points)

    sdf_points, sdf_values = sample_sdf_near_surface(obj_mesh, number_of_points=500000)
    trans, scale = compute_unit_sphere_transform(obj_mesh)

    pos_idx = np.where(sdf_values >= 0)[0]
    neg_idx = np.where(sdf_values < 0)[0]

    sdf_pos = np.concatenate([sdf_points[pos_idx], sdf_values[pos_idx][:, None]], axis=1)
    sdf_neg = np.concatenate([sdf_points[neg_idx], sdf_values[neg_idx][:, None]], axis=1)

    np.savez(os.path.join(output_sdf_dir, obj_name + ".npz"), pos=sdf_pos, neg=sdf_neg, scale=scale, trans=trans)
