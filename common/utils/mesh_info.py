import argparse
import os
import numpy as np
import trimesh
from trimesh.primitives import Box
from trimesh import bounds
from transforms3d import quaternions


def create_collision_mesh(obj_path):
    obj_name = obj_path.split('/')[-2]
    mesh = trimesh.load(obj_path)
    faces = mesh.faces

    if obj_name in ['019_pitcher_base', '037_scissors']:
        # Get the principal components of the mesh
        points, _ = trimesh.sample.sample_surface_even(mesh, 50000)
        points_2d = points[:, :2]

        C = np.cov(points_2d.T)
        eigvals, eigvecs = np.linalg.eig(C)
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Set the x and y axes to the first and second principal components, respectively
        x_axis = np.array([eigvecs[0, 0], eigvecs[1, 0], 0])
        z_axis = np.array([0, 0, 1])
        y_axis = np.cross(z_axis, x_axis)
    else:
        box = mesh.bounding_box_oriented
        index = np.where(box.vertices[:, 2] > 0)
        selected_vertices = box.vertices[index]

        edge_1 = np.linalg.norm(selected_vertices[0, :2] - selected_vertices[1, :2])
        edge_2 = np.linalg.norm(selected_vertices[0, :2] - selected_vertices[2, :2])
        edge_3 = np.linalg.norm(selected_vertices[1, :2] - selected_vertices[2, :2])

        if edge_1 < np.max([edge_1, edge_2, edge_3]):
            x_axis_ori = selected_vertices[0] - selected_vertices[1]
        else:
            x_axis_ori = selected_vertices[1] - selected_vertices[2]

        x_axis = np.array([x_axis_ori[0], x_axis_ori[1], 0])
        z_axis = np.array([0, 0, 1])
        y_axis = np.cross(z_axis, x_axis)

    # Normalize the axes
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    
    # Create a rotation matrix to align with the principal axes
    trans_matrix = np.array([
        [x_axis[0], y_axis[0], z_axis[0], 0],
        [x_axis[1], y_axis[1], z_axis[1], 0],
        [x_axis[2], y_axis[2], z_axis[2], 0],
        [0, 0, 0, 1]])

    quat = quaternions.mat2quat(trans_matrix[:3, :3].T)
    new_quat = [f'{x:.4f}' for x in quat]
    output_quat = ', '.join(new_quat)
    new_vertices = (trans_matrix[:3, :3].T @ mesh.vertices.transpose(1, 0)).transpose(1, 0)
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=faces)

    new_mesh.export(f'cmesh/{obj_name}.obj')
    
    # calculate the 3D dimensional size of the mesh
    min_coords = np.min(new_mesh.vertices, axis=0)
    max_coords = np.max(new_mesh.vertices, axis=0)
    size = max_coords - min_coords
    new_size = [f'{x:.4f}' for x in size]
    output_size = ', '.join(new_size)
    print(f'{obj_name}: {output_size} || {output_quat}')

if __name__ == '__main__':
    # Create the collision mesh and write it to a file
    objects = ['004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', '010_potted_meat_can', '011_banana', '025_mug', '051_large_clamp', '061_foam_brick']
    classes = ['008_pudding_box', '002_master_chef_can',    '003_cracker_box',    '004_sugar_box',    '005_tomato_soup_can',    '006_mustard_bottle',    '007_tuna_fish_can',    '009_gelatin_box',    '010_potted_meat_can',    '011_banana',    '019_pitcher_base',    '021_bleach_cleanser',    '024_bowl',    '025_mug',    '035_power_drill',    '036_wood_block',    '037_scissors',    '040_large_marker',    '051_large_clamp',    '052_extra_large_clamp',    '061_foam_brick']
    ycb_model_path = '/home/zerui/Desktop/models/'
    for obj_name in classes:
        if obj_name not in objects:
            visual_path = f'/Users/chenzerui/workspace/code/dex_sim/hand_imitation/env/models/assets/ycb/visual/{obj_name}'
            create_collision_mesh(os.path.join(visual_path, 'textured_simple.stl'))

