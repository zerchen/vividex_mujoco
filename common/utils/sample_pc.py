import numpy as np
import trimesh
import os

# output_dir = "../../demonstrations/rest_pc/"
# mesh_root_dir = '../../hand_imitation/env/models/assets/ycb/visual/'
# for filename in os.listdir(mesh_root_dir):
    # obj_mesh = trimesh.load(os.path.join(mesh_root_dir, filename, 'textured_simple.stl'))
    # points, faces = trimesh.sample.sample_surface(obj_mesh, 512)
    # normals = obj_mesh.face_normals[faces]
    # point_info = np.concatenate([points, normals], 1)
    # box_info = obj_mesh.bounding_box.bounds.copy()

    # obj_name = '_'.join(filename.split('_')[1:])
    # np.savez(os.path.join(output_dir, obj_name + ".npz"), pc=point_info, box=box_info)

output_dir = "../../demonstrations/rest_pc/"
mesh_root_dir = '../../datasets/hoi4d/HOI4D_CAD_Model_for_release/rigid/'
for category in os.listdir(mesh_root_dir):
    for filename in os.listdir(os.path.join(mesh_root_dir, category)):
        if category == "Chair":
            continue
        obj_mesh = trimesh.load(os.path.join(mesh_root_dir, category, filename))
        points, faces = trimesh.sample.sample_surface(obj_mesh, 512)
        normals = obj_mesh.face_normals[faces]
        point_info = np.concatenate([points, normals], 1)
        box_info = obj_mesh.bounding_box.bounds.copy()

        obj_name = 'hoi4d_' + category + '-' + str(int(filename.split('.')[0])).zfill(4)
        np.savez(os.path.join(output_dir, obj_name + ".npz"), pc=point_info, box=box_info)

# output_dir = "../../demonstrations/rest_pc/"
# category = "shapenet_remote"
# mesh_root_dir = f'../../hand_imitation/env/models/assets/{category}/visual/'
# for filename in os.listdir(mesh_root_dir):
    # obj_mesh = trimesh.load(os.path.join(mesh_root_dir, filename, 'model_transform_scaled.stl'))
    # points, faces = trimesh.sample.sample_surface(obj_mesh, 512)
    # normals = obj_mesh.face_normals[faces]
    # point_info = np.concatenate([points, normals], 1)
    # box_info = obj_mesh.bounding_box.bounds.copy()

    # obj_name = category + '-' + filename
    # np.savez(os.path.join(output_dir, obj_name + ".npz"), pc=point_info, box=box_info)
