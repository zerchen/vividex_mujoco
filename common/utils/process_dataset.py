import os
import cv2
import sys
import torch
import yaml
import pickle
import json
import trimesh
import argparse
import numpy as np
import open3d as o3d
import transforms3d
from tqdm import tqdm
from natsort import natsorted
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree as KDTree
from moviepy.editor import ImageSequenceClip

sys.path.insert(0, '../')
from mano.manolayer import ManoLayer
sys.path.insert(0, '../../')
from datasets.dexycb.toolkit.dex_ycb import _YCB_CLASSES
from datasets.dexycb.toolkit.factory import get_dataset
from datasets.dexycb.toolkit.layers.ycb_layer import dcm2rv, rv2dcm


def convert(depth_path, mask_path):
    # background
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    assert mask.shape == (1080, 1920, 3)

    s = np.sum(mask, axis=2)
    depth_b = depth.copy()
    depth_b[s > 0] = 0
    cv2.imwrite(depth_path.split('.')[0] + '_bg.png', depth_b)
    depth_b = o3d.io.read_image(depth_path.split('.')[0] + '_bg.png')
    
    return depth_b


def vis_scene(pc_scene, hand_verts, obj_pose, obj_name):
    mano_faces = np.load('../common/mano/assets/closed_fmano.npy')

    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.vertices = o3d.utility.Vector3dVector(np.copy(hand_verts))
    hand_mesh.triangles = o3d.utility.Vector3iVector(np.copy(mano_faces))
    hand_mesh.paint_uniform_color([0.6, 0.1, 0.5])

    category_name = obj_name.split('-')[0]
    instance_id = obj_name.split('-')[1].zfill(3)
    obj_mesh_path = f"../datasets/hoi4d/HOI4D_CAD_Model_for_release/rigid/{category_name}/{instance_id}.obj"
    obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
    obj_mesh = obj_mesh.transform(obj_pose)
    obj_mesh.paint_uniform_color([0.1, 0.6, 0.5])

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pc_scene, hand_mesh, obj_mesh, coord_frame])


def read_pcd(filepath, vis=False):
    seq_id = '-'.join(filepath.split('/')[4:11])
    scene_id = seq_id.split('-')[4]
    with open(filepath, "r") as f:
        lines = [line.strip().split(" ") for line in f.readlines()]
    
    cat_name = filepath.split('/')[6]
    pc = lines[10:]
    pc_label = np.array(pc)

    if scene_id in ["S59", "S60", "S61", "S79", "S270", "S272"]:
        pc_index = np.where(pc_label[:, 4].astype(np.uint8) == 38)[0]
    else:
        pc_index = np.where(pc_label[:, 4].astype(np.uint8) == 3)[0]
        if len(pc_index) == 0:
            pc_index = np.where(pc_label[:, 4].astype(np.uint8) == 19)[0]
        if len(pc_index) == 0:
            pc_index = np.where(pc_label[:, 4].astype(np.uint8) == 32)[0]
        if len(pc_index) == 0:
            pc_index = np.where(pc_label[:, 4].astype(np.uint8) == 33)[0]
        if len(pc_index) == 0:
            pc_index = np.where(pc_label[:, 4].astype(np.uint8) == 34)[0]
        if len(pc_index) == 0:
            pc_index = np.where(pc_label[:, 4].astype(np.uint8) == 38)[0]

    points = np.asarray(pc_label[:, :3], dtype=np.float32)[pc_index]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.01, ransac_n=5, num_iterations=2000)
    
    if plane_model[1] * plane_model[2] < 0:
        inlier_cloud = point_cloud.select_by_index(inliers, invert=True)
        plane_model, inliers = inlier_cloud.segment_plane(distance_threshold=0.01, ransac_n=5, num_iterations=2000)
        inlier_cloud = inlier_cloud.select_by_index(inliers)
    else:
        inlier_cloud = point_cloud.select_by_index(inliers)
    mean, covariance = inlier_cloud.compute_mean_and_covariance()
    _, principal_axis = np.linalg.eig(covariance)
    x_axis = principal_axis[:, np.where(np.abs(principal_axis[0]) == np.max(np.abs(principal_axis[0])))[0]].reshape(-1)
    x_axis = np.sign(x_axis[0]) * x_axis
    
    inlier_cloud = inlier_cloud.translate(-mean)
    
    axis = np.cross(np.array([1, 0, 0]), x_axis)
    angle = np.dot(np.array([1, 0, 0]), x_axis)
    axisang = angle * axis
    x_mat = inlier_cloud.get_rotation_matrix_from_axis_angle(axisang)
    
    z_axis = -np.sign(plane_model[1]) * plane_model[:3]
    if np.dot(z_axis, np.array([0, 0, 1])) > 0:
        z_axis = -1 * z_axis
    y_axis = np.cross(z_axis, x_axis)
    rot_mat = np.eye(3)
    rot_mat[:, 0] = x_axis
    rot_mat[:, 1] = y_axis
    rot_mat[:, 2] = z_axis
    inlier_cloud = inlier_cloud.rotate(rot_mat.T)

    trans_matrix = np.eye(4)
    trans_matrix[:3, 3] = -mean
    rot_matrix = np.eye(4)
    rot_matrix[:3, :3] = rot_mat.T
    camera_2_world = rot_matrix @ trans_matrix

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    points = np.asarray(pc_label[:, :3], dtype=np.float32)
    all_pc = o3d.geometry.PointCloud()
    all_pc.points = o3d.utility.Vector3dVector(points)
    all_pc = all_pc.transform(camera_2_world)
    if vis:
        o3d.visualization.draw_geometries([all_pc, coord_frame])

    return camera_2_world, all_pc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    np.set_printoptions(precision=4)

    seq_dict = {}
    if args.dataset == 'ycb':
        dataset = get_dataset('s4_train')
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            if sample['mano_side'] == 'right':
                img_path = sample['color_file']
                subject_id = img_path.split('data/')[-1].split('/')[0]
                video_id = img_path.split('data/')[-1].split('/')[1]
                camera_id = img_path.split('data/')[-1].split('/')[2]
                frame_idx = int(img_path.split('data/')[-1].split('/')[-1].split('.')[0].split('color_')[-1])

                meta_file_path = os.path.join(dataset._data_dir, subject_id, video_id, 'meta.yml')
                with open(meta_file_path, 'r') as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)
                extr_id = meta['extrinsics']
                extrinsics_path = os.path.join(dataset._calib_dir, f'extrinsics_{extr_id}', 'extrinsics.yml')
                with open(extrinsics_path, 'r') as f:
                    extrinsics = yaml.load(f, Loader=yaml.FullLoader)
                extr_table = np.array(extrinsics['extrinsics']['apriltag'], dtype=np.float32).reshape((3, 4))
                table_matrix = torch.eye(4)
                table_matrix[:3, :] = torch.from_numpy(extr_table)

                extr_cam = np.array(extrinsics['extrinsics'][camera_id], dtype=np.float32).reshape((3, 4))
                cam_matrix = torch.eye(4)
                cam_matrix[:3, :] = torch.from_numpy(extr_cam)
                update_cam_matrix = torch.inverse(table_matrix) @ cam_matrix

                # center_matrix aims to move the table coordinate system to the center of the table
                center_matrix = torch.eye(4)
                center_matrix[:3, 3] = torch.Tensor([-0.6, -0.3, 0])
                direction_matrix = torch.eye(4)
                direction_matrix[0, 0] = -1.
                direction_matrix[1, 1] = -1.
                update_cam_matrix = direction_matrix @ center_matrix @ update_cam_matrix

                grasp_obj_id = sample['ycb_ids'][sample['ycb_grasp_ind']]
                object_name = _YCB_CLASSES[grasp_obj_id]

                seq_id = subject_id + '/' + video_id
                if seq_id not in seq_dict:
                    seq_dict[seq_id] = dict()
                    seq_dict[seq_id]['SIM_SUBSTEPS'] = 10
                    seq_dict[seq_id]['DATA_SUBSTEPS'] = 1
                    seq_dict[seq_id]['length'] = 0
                    seq_dict[seq_id]['img_path'] = []
                    seq_dict[seq_id]['object_translation'] = []
                    seq_dict[seq_id]['object_orientation'] = []
                    seq_dict[seq_id]['hand_joint'] = []
                    seq_dict[seq_id]['s_0'] = dict()
                    seq_dict[seq_id]['s_0']['pregrasp'] = dict()
                    seq_dict[seq_id]['s_0']['pregrasp']['position'] = []

                label = np.load(sample['label_file'])
                if np.all(label['joint_3d'][0] == -1):
                    continue

                mano_layer = ManoLayer(flat_hand_mean=False, ncomps=45, side=sample['mano_side'], mano_root='../mano/assets/', use_pca=True)
                betas = torch.tensor(sample['mano_betas'], dtype=torch.float32).unsqueeze(0)
                # convert hand pose to the table frame
                v = torch.matmul(mano_layer.th_shapedirs, betas.transpose(0, 1)).permute(2, 0, 1) + mano_layer.th_v_template
                root_trans = torch.matmul(mano_layer.th_J_regressor[0], v).cpu().numpy()
                mano_pose = label['pose_m'].reshape((-1, 51))
                wrist_pose = mano_pose[:, np.r_[0:3, 48:51]]
                wrist_pose[:, 3:] += root_trans
                hand_pose = torch.eye(4)
                hand_pose[:3, :3] = torch.from_numpy(Rot.from_rotvec(wrist_pose[0, :3]).as_matrix())
                hand_pose[:3, 3] = torch.from_numpy(wrist_pose[0, 3:])
                hand_pose = (update_cam_matrix @ hand_pose).numpy()
                wrist_rot = Rot.from_matrix(hand_pose[:3, :3]).as_rotvec()
                wrist_trans = hand_pose[:3, 3] - root_trans[0]
                wrist_pose = np.hstack([wrist_rot, wrist_trans])
                mano_pose[:, np.r_[0:3, 48:51]] = wrist_pose
                mano_nowrist_pose = mano_pose.copy()
                mano_nowrist_pose[:, 0:3] = 0.0
                hand_verts_3d, hand_joints_3d, _, hand_global_trans, _ = mano_layer(torch.from_numpy(mano_pose[:, 0:48]), betas, torch.from_numpy(mano_pose[:, 48:51]))

                hand_verts_3d = hand_verts_3d[0].numpy()
                hand_joints_3d = hand_joints_3d[0].numpy()

                obj_pose = torch.eye(4)
                obj_pose[:3, :] = torch.from_numpy(label['pose_y'][sample['ycb_grasp_ind']])
                world_obj_pose = (update_cam_matrix @ obj_pose).numpy()
                grasp_obj_id = sample['ycb_ids'][sample['ycb_grasp_ind']]
                obj_rest_verts = trimesh.load(dataset.obj_file[grasp_obj_id], process=False).vertices
                homo_obj_rest_verts = np.ones((obj_rest_verts.shape[0], 4), dtype=np.float32)
                homo_obj_rest_verts[:, :3] = obj_rest_verts
                homo_obj_verts = (world_obj_pose @ homo_obj_rest_verts.transpose(1, 0)).transpose(1, 0)
                obj_verts = homo_obj_verts[:, :3] / homo_obj_verts[:, [3]]

                hand_points_kd_tree = KDTree(hand_verts_3d)
                obj2hand_distances, _ = hand_points_kd_tree.query(obj_verts)
                if len(seq_dict[seq_id]['s_0']['pregrasp']['position']) == 0 and obj2hand_distances.min() <= 0.005:
                    if np.min(obj_verts[:, 2]) > 0.05:
                        continue
                    seq_dict[seq_id]['object_name'] = object_name
                    seq_dict[seq_id]['s_0']['pregrasp']['position'] = hand_joints_3d.tolist()
                    seq_dict[seq_id]['s_0']['pregrasp']['vertice'] = hand_verts_3d.tolist()

                if len(seq_dict[seq_id]['s_0']['pregrasp']['position']) > 0:
                    seq_dict[seq_id]['length'] += 1
                    seq_dict[seq_id]['img_path'].append(img_path)
                    seq_dict[seq_id]['hand_joint'].append(hand_joints_3d.tolist())
                    seq_dict[seq_id]['object_translation'].append(world_obj_pose[:3, 3].tolist())
                    seq_dict[seq_id]['object_orientation'].append(list(Quaternion(matrix=world_obj_pose[:3, :3])))

        output_dir = f"../../hand_imitation/env/models/assets/trajectories/{args.dataset}"
        output_videos_dir = f"../../hand_imitation/env/models/assets/trajectories/goal_videos/{args.dataset}"
        object_mesh_dir = f"../../hand_imitation/env/models/assets/objects/ycb/"
        for seq_name in seq_dict.keys():
            seq_suffix = seq_name.replace('/', '-')

            if len(seq_dict[seq_name]['img_path']) == 0:
                continue

            length = seq_dict[seq_name]['length']
            object_name = seq_dict[seq_name]['object_name']
            hand_joint = np.array(seq_dict[seq_name]['hand_joint'])
            object_translation = np.array(seq_dict[seq_name]['object_translation'])
            object_orientation = np.array(seq_dict[seq_name]['object_orientation'])
            sim_steps = seq_dict[seq_name]['SIM_SUBSTEPS']
            data_steps = seq_dict[seq_name]['DATA_SUBSTEPS']
            s_0 = seq_dict[seq_name]['s_0']

            pregrasp_obj_pose = np.eye(4, dtype=np.float32)
            object_mesh = trimesh.load(os.path.join(object_mesh_dir, object_name, 'textured.obj'), process=False)
            obj_verts = object_mesh.vertices.copy()
            obj_faces = object_mesh.faces.copy()
            obj_trans_verts = (Quaternion(object_orientation[0]).rotation_matrix @ obj_verts.transpose(1, 0)).transpose(1, 0) + object_translation[0]
            object_trans_mesh = trimesh.Trimesh(vertices=obj_trans_verts, faces=obj_faces)
            object_trans_mesh.export(os.path.join(output_dir, f'{args.dataset}-{object_name}-{seq_suffix}_obj.obj'))

            hand_verts = np.array(s_0['pregrasp']['vertice'])
            hand_faces = np.load('../mano/assets/closed_fmano.npy')
            hand_trans_mesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces)
            hand_trans_mesh.export(os.path.join(output_dir, f'{args.dataset}-{object_name}-{seq_suffix}_hand.obj'))

            video_filename = os.path.join(output_videos_dir, f'{args.dataset}-{object_name}-{seq_suffix}.gif')
            cl = ImageSequenceClip(seq_dict[seq_name]['img_path'], fps=30)
            cl.write_gif(video_filename, fps=30)

            output_path = os.path.join(output_dir, f'{args.dataset}-{object_name}-{seq_suffix}.npz')
            np.savez(output_path, hand_joint=hand_joint, object_translation=object_translation, object_orientation=object_orientation, length=length, object_name=object_name, s_0=s_0, SIM_SUBSTEPS=sim_steps, DATA_SUBSTEPS=data_steps)


if __name__ == '__main__':
    main()
