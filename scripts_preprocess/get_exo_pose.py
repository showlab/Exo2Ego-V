from torch.utils.data import Dataset
import pandas as pd
from scipy.spatial.transform import Rotation
import json

from projectaria_tools.core import mps
import pdb
import numpy as np
import torch
import os
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.stream_id import StreamId

def construct_extrinsic_matrix(rotation_matrix, translation_vector):
        # [R | t]
        extrinsic_matrix = np.vstack((np.hstack((rotation_matrix, translation_vector.reshape(-1, 1))),
                                    np.array([0, 0, 0, 1])))
        return extrinsic_matrix

def quaternion_to_rotation_matrix(quaternion):
    rotation = Rotation.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    return rotation_matrix

def get_exo_pose(pose_file):
        poses = pd.read_csv(pose_file)
        all_poses = []
        all_focals = []
        all_c = []
        num = len(poses)
        for i in range(num):
            rotation = poses.loc[i, ['qx_world_cam', 'qy_world_cam', 'qz_world_cam', 'qw_world_cam']].values
            translation = poses.loc[i, ['tx_world_cam', 'ty_world_cam', 'tz_world_cam']].astype(float).values
            rotation_matrix = quaternion_to_rotation_matrix(rotation)
            pose = construct_extrinsic_matrix(rotation_matrix,translation)
            pose = torch.from_numpy(np.array(pose)).float()
            focal =  poses.loc[i,['intrinsics_0','intrinsics_1']].astype(float).values
            focal = torch.from_numpy(focal).float()
            c = poses.loc[i,['intrinsics_2','intrinsics_3']].astype(float).values
            c = torch.from_numpy(c).float()
            all_poses.append(pose)
            all_focals.append(focal)
            all_c.append(c)
        if all_poses!=[]:
            all_poses = torch.stack(all_poses)
            focal = torch.stack(all_focals)
            c = torch.stack(all_c)
            return all_poses,focal,c
        else:
            return None,None,None

takes_path = '"Your Data Folder"/takes'
names = ["category"] # choose a category
exo = {}
exo_path = '"Your Output Folder"/exo_extrinsic.pt'
camera_base_folder = '"Your Data Folder"/takes'
num = 0
for name in names:
    for file_name in os.listdir(takes_path):
        if name in file_name:
            num+=1
            exo[file_name] = {}
            ss =  file_name.rsplit('_', 1)[0]
            pose_path = os.path.join(camera_base_folder,file_name,"trajectory","gopro_calibs.csv")
            if not os.path.exists(pose_path):
               continue
            exo_poses, all_focals,all_c = get_exo_pose(pose_path)
            exo[file_name]['poses'] = exo_poses
            exo[file_name]['all_focals'] = all_focals
            exo[file_name]['all_c'] = all_c
            print(file_name)
torch.save(exo, exo_path)

def get_exo_intrinsic(static_cameras_path,width=480,height=270):
    static_cameras = mps.read_static_camera_calibrations(static_cameras_path)
    exo_cams = []
    for i in range(len(static_cameras)):
        exo_cams.append(calibration.CameraCalibration('camera-rgb', calibration.CameraModelType.KANNALA_BRANDT_K3, np.expand_dims(static_cameras[i].intrinsics,-1), 
                                        static_cameras[i].transform_world_cam, static_cameras[i].width, static_cameras[i].height, None, 2*np.pi, 'None'))
    cam_unproj_map = []
    parameters = []
    for exo_cam in exo_cams:
        X,Y = np.meshgrid(
            np.arange(width, dtype=np.float32) ,
            np.arange(height, dtype=np.float32) ,
        )
        proj = np.stack((X, Y), axis=-1)
        unproj = np.zeros((height, width, 3))
        for i in range(height):
            for j in range(width):
                unproj[i, j, :] = exo_cam.unproject(proj[i, j, :].reshape((2, 1))*(exo_cam.get_image_size()[0]/width))
        unproj = torch.from_numpy(unproj).to("cpu").to(torch.float32)
        unproj[:, :, 1:3] *= -1
        unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1)
        cam_unproj_map.append(unproj)
        parameters.append(torch.from_numpy(exo_cam.projection_params()))
    if parameters !=[]:
        parameters = torch.stack(parameters)
        cam_unproj_map = torch.stack(cam_unproj_map)
        return parameters, cam_unproj_map
    else:
        return None,None

unproj={}
exo_cams_all={}
exo_path = '"Your Output Folder"/unproj_exo.pt'
exo_cams_path = '"Your Output Folder"/exo_parameters.pt'
for scene in os.listdir(takes_path):
    for name in names:
        if name in scene:
            ss =  scene.rsplit('_', 1)[0]
            print(scene)
            path = os.path.join('"Your Data Folder"/takes',scene,"trajectory","gopro_calibs.csv")
            if not os.path.exists(path):
                continue
            exo_cams_all[scene] ,unproj[scene] = get_exo_intrinsic(path)

torch.save(unproj, exo_path)
torch.save(exo_cams_all, exo_cams_path)
