import argparse
import random
import os

import accelerate
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from torch.utils.data import Dataset
import pandas as pd
from scipy.spatial.transform import Rotation
import json

from projectaria_tools.core import mps
import pdb
import pickle

def choose_folder(base_folder,folders):

    folder_file_counts = [len(os.listdir(os.path.join( base_folder,folder))) for folder in folders]
    total_files = sum(folder_file_counts)
    probabilities = [count / total_files for count in folder_file_counts]

    return probabilities


class EgoExo4Ddataset(Dataset):
    def __init__(self, args,num_samples=100000, sample_frames=25):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        self.num_samples = num_samples
        # Define the path to the folder containing video frames
        self.base_folder = args.base_dir
        if args.scene_list is None:
            self.scene_folders = os.listdir(self.base_folder)
        else:
            self.scene_folders = args.scene_list

        self.channels = 3
        self.sample_frames = sample_frames
        self.exo_width = args.exo_width
        self.exo_height = args.exo_height
        self.ego_width = args.ego_width
        self.ego_height = args.ego_height
        self.folder_ego = args.ego_dir
        self.folder_exo = args.exo_dir
        self.probabilities = {}
        self.condition_ego = args.condition_ego
        self.cat_ego = args.cat_ego
        self.pose_dir = args.pose_dir
        self.data_list = args.data_list
        self.validation_data_list = args.validation_data_list
        self.validation_dir = args.validation_dir        
        for scene_folder in  self.scene_folders:
            base_folder = os.path.join(self.base_folder,scene_folder,"ego")
            if args.data_list is None:
                folders = os.listdir(base_folder)
            else:
                folders = args.data_list
            probabilities = choose_folder(base_folder,folders)
            self.probabilities[scene_folder] = probabilities
        self.original_width = 3840
        self.exo_dict = torch.load(os.path.join(args.pose_dir,"exo_extrinsic.pt"))
        if args.train_dict is not None:
            with open(args.train_dict, 'rb') as file:
                self.train_dict = pickle.load(file)
        if args.test_dict is not None:
            with open(args.test_dict, 'rb') as file:
                self.test_dict = pickle.load(file)

    def get_validation(self,args,accelerator):

        all_validation_values = {}
        validation_files=[]
        while(1):
            category = random.choice(list(self.test_dict.keys()))
            if self.test_dict[category] !={}:
                scene_folder = random.choice(list(self.test_dict[category].keys()))
                folders = self.test_dict[category][scene_folder]
                if folders !='all' and folders!=[]:
                    break

        chosen_folders = [random.choice(folders)]
        start_idxes = [0 for i in range(len(folders))]
        all_validation_values_ego = []

        exo_poses, all_focals,all_c = self.get_exo_pose2(scene_folder)
        
        all_ego_poses = []
        ego_images = []

        chosen_folders = ['0']

        for idx, chosen_folder in enumerate(chosen_folders):
            frames = os.listdir(os.path.join(args.base_dir,scene_folder,"ego",chosen_folder))
            frames = sorted(frames, key=lambda x: int(x.split('_')[1].split('.')[0]) if x.split('_')[1].split('.')[0].isdigit() else float('inf'))
            start_idx  = start_idxes[idx]
            selected_frames = frames[start_idx:start_idx + args.num_frames]
            all_ego_poses.append(self.get_ego_pose(scene_folder, chosen_folder,selected_frames))
            validation_values = torch.empty((args.num_frames, 3, args.ego_height, args.ego_width))
            for i, frame_name in enumerate(selected_frames):
                frame_output = os.path.join(args.base_dir,scene_folder,"ego", chosen_folder, frame_name)
                validation_files.append(frame_output)
                with Image.open(frame_output) as img:
                    # Resize the image and convert it to a tensor
                    img_resized = img.resize((args.ego_width, args.ego_height))
                    if i ==0:
                        ego_image = img.resize((args.ego_width//8, args.ego_height//8)) 
                        ego_image =  torch.from_numpy(np.array(ego_image)).float()
                        ego_image = ego_image / 127.5 - 1
                        ego_image = ego_image.permute(2, 0, 1)
                        ego_images.append(ego_image)
                    img_tensor = torch.from_numpy(np.array(img_resized)).float()

                    # Normalize the image by scaling pixel values to [-1, 1]
                    img_normalized = img_tensor / 127.5 - 1
                    img_normalized = img_normalized.permute(2, 0, 1)  # For RGB images

                    validation_values[i] = img_normalized
            all_validation_values_ego.append(validation_values.unsqueeze(0).to(accelerator.device, non_blocking=True))
        all_validation_values['ego'] = all_validation_values_ego
        all_ego_poses = torch.stack(all_ego_poses)
        ego_images = torch.stack(ego_images)

        folder_name = os.path.join(args.base_dir,scene_folder,"exo")
        all_validation_values['exo'] = []
        for idx, chosen_folder in enumerate(chosen_folders):
            all_validation_values_exo = []
            for i in range(4):
                if i==0:
                    folder_name = os.path.join(args.base_dir,scene_folder,"exo")
                else:
                    folder_name = os.path.join(args.base_dir,scene_folder,"exo"+str(i+1))
                frames = os.listdir(os.path.join(folder_name,chosen_folder))
                frames = sorted(frames, key=lambda x: int(x.split('_')[1].split('.')[0]) if x.split('_')[1].split('.')[0].isdigit() else float('inf'))

                start_idx  = start_idxes[idx]
                selected_frames = frames[start_idx:start_idx + args.num_frames]
                validation_values = torch.empty((args.num_frames, 3, args.exo_height, args.exo_width))
                for i, frame_name in enumerate(selected_frames):
                    frame_output = os.path.join(folder_name, chosen_folder, frame_name)
                    validation_files.append(frame_output)
                    with Image.open(frame_output) as img:
                        # Resize the image and convert it to a tensor
                        img_resized = img.resize((args.exo_width, args.exo_height)) # hard code here
                        img_tensor = torch.from_numpy(np.array(img_resized)).float()

                        # Normalize the image by scaling pixel values to [-1, 1]
                        img_normalized = img_tensor / 127.5 - 1
                        img_normalized = img_normalized.permute(2, 0, 1)  # For RGB images
                        validation_values[i] = img_normalized
                all_validation_values_exo.append(validation_values.unsqueeze(0).to(accelerator.device, non_blocking=True))
            all_validation_values['exo'].append(all_validation_values_exo)
        all_focals/=(self.original_width/self.exo_width)
        all_c /=(self.original_width/self.exo_width)

        all_ego_poses[:, :, :, 1:3] *= -1
        exo_poses[:, :, 1:3] *= -1

        rel_poses =  torch.matmul(torch.linalg.inv(all_ego_poses[0]).unsqueeze(1), exo_poses)

        result = {
            "all_validation_values":all_validation_values,
            "validation_files": validation_files,
            "ego_poses":all_ego_poses,
            "exo_poses":exo_poses,
            'scene': scene_folder,
            'focal': all_focals,
            'c': all_c,
            'ego_images':ego_images,
            "rel_poses": rel_poses.squeeze(),
        }
        return result

    def __len__(self):
        return self.num_samples


    def construct_extrinsic_matrix(self,rotation_matrix, translation_vector):
        # Construct [R | t]
        extrinsic_matrix = np.vstack((np.hstack((rotation_matrix, translation_vector.reshape(-1, 1))),
                                    np.array([0, 0, 0, 1])))
        return extrinsic_matrix


    def quaternion_to_rotation_matrix(self,quaternion):
        rotation = Rotation.from_quat(quaternion)
        rotation_matrix = rotation.as_matrix()
        return rotation_matrix
    
    def get_aria_start_end_sec(self, timesync_path, start_idx, end_idx):
        timesync = pd.read_csv(timesync_path)
        aria_rgb_col = next(
            col
            for col in timesync.columns
            if "aria" in col.lower() and "214-1" in col and "capture_timestamp_ns" in col
        )
        return (
            timesync.iloc[start_idx][aria_rgb_col] * 1e-9,
            timesync.iloc[end_idx][aria_rgb_col] * 1e-9,
        )

    def get_exo_pose(self,pose_file):
        poses = pd.read_csv(pose_file)
        all_poses = []
        all_focals = []
        all_c = []
        num = len(poses)
        for i in range(num): 
            rotation = poses.loc[i, ['qx_world_cam', 'qy_world_cam', 'qz_world_cam', 'qw_world_cam']].values
            translation = poses.loc[i, ['tx_world_cam', 'ty_world_cam', 'tz_world_cam']].astype(float).values
            rotation_matrix = self.quaternion_to_rotation_matrix(rotation)
            pose = self.construct_extrinsic_matrix(rotation_matrix,translation)
            pose = torch.from_numpy(np.array(pose)).float()
            focal =  poses.loc[i,['intrinsics_0','intrinsics_1']].astype(float).values
            focal = torch.from_numpy(focal).float()
            c = poses.loc[i,['intrinsics_2','intrinsics_3']].astype(float).values
            c = torch.from_numpy(c).float()
            all_poses.append(pose)
            all_focals.append(focal)
            all_c.append(c)
        all_poses = torch.stack(all_poses)
        focal = torch.stack(all_focals)
        c = torch.stack(all_c)

        return all_poses,focal,c
    
    def get_exo_pose2(self,scene):

        all_poses = self.exo_dict[scene]['poses'].clone()
        focal = self.exo_dict[scene]['all_focals'][0].clone()
        c = self.exo_dict[scene]['all_c'][0].clone()
        return all_poses,focal,c
    
    def get_ego_pose(self,scene,choosen_folder,ego_files):

        path = os.path.join(self.pose_dir,scene,"pose.json")
        poses = []
        if not os.path.exists(path):
            return None
        with open(path, 'r') as file:
            data = json.load(file)
            for ego_file in ego_files:
                poses.append(data[ego_file])
        poses = torch.tensor(poses)
        return poses
        
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        # Randomly select a folder (representing a video) from the base folder
        i=0
        while(1):
            i+=1
            category = random.choice(list(self.train_dict.keys()))
            scene_folder = random.choice(list(self.train_dict[category].keys()))
            exo_poses, all_focals,all_c = self.get_exo_pose2(scene_folder)
            ego_folder = os.path.join(self.base_folder, scene_folder,"ego")
            exo_folder = os.path.join(self.base_folder, scene_folder,"exo")
            if exo_poses.shape[0]!=4:
                continue
            
            folders = self.train_dict[category][scene_folder]
            if folders == []:
                pdb.set_trace()
            if folders =="all":
                folders = os.listdir(ego_folder)
            while(1):
                chosen_folder = random.choices(folders)[0]
                frames = os.listdir(os.path.join(ego_folder,chosen_folder))
                if len(frames) >= self.sample_frames:
                    break

            folder_path = os.path.join(ego_folder, chosen_folder)
            frames = os.listdir(folder_path)
            # Sort the frames by name
            frames = sorted(frames, key=lambda x: int(x.split('_')[1].split('.')[0]) if x.split('_')[1].split('.')[0].isdigit() else float('inf'))
            # Ensure the selected folder has at least `sample_frames`` frames
            if len(frames) < self.sample_frames:
                raise ValueError(
                    f"The selected folder '{chosen_folder}' contains fewer than `{self.sample_frames}` frames.")
            # Randomly select a start index for frame sequence
            start_idx = random.randint(0, len(frames) - self.sample_frames)
            selected_frames = frames[start_idx:start_idx + self.sample_frames]
            ego_poses = self.get_ego_pose(scene_folder, chosen_folder,selected_frames)
            if ego_poses is None:
                continue

            # Initialize a tensor to store the pixel values
            pixel_values = torch.empty((self.sample_frames, self.channels, self.ego_height, self.ego_width))
            pixel_files = []
            ego_images = torch.empty((self.sample_frames, self.channels, self.ego_height//8, self.ego_width//8))
            # Load and process each frame
            for i, frame_name in enumerate(selected_frames):
                frame_path = os.path.join(folder_path, frame_name)
                pixel_files.append(frame_path)

                with Image.open(frame_path) as img:
                    # Resize the image and convert it to a tensor
                    img_resized = img.resize((self.ego_width, self.ego_height))
                    ego_image = img.resize((self.ego_width//8, self.ego_height//8))
                    ego_image =  torch.from_numpy(np.array(ego_image)).float()
                    ego_image = ego_image / 127.5 - 1
                    ego_image = ego_image.permute(2, 0, 1)
                    img_tensor = torch.from_numpy(np.array(img_resized)).float()
                    # Normalize the image by scaling pixel values to [-1, 1]
                    img_normalized = img_tensor / 127.5 - 1
                    img_normalized = img_normalized.permute(2, 0, 1) 
                    pixel_values[i] = img_normalized
                    ego_images[i] = ego_image
            
            condition_values = []       
            condition_files = []
            for j in range(4):
                condition_value = torch.empty((self.sample_frames, self.channels, self.exo_height, self.exo_width))

                for i, frame_name in enumerate(selected_frames):
                    if j ==0:
                        frame_output = os.path.join(exo_folder, chosen_folder, frame_name)
                    else:
                        frame_output = os.path.join(exo_folder+str(j+1), chosen_folder,frame_name)
                    condition_files.append(frame_output)
                    with Image.open(frame_output) as img:
                        # Resize the image and convert it to a tensor
                        img_resized = img.resize((self.exo_width, self.exo_height)) # hard code here
                        img_tensor = torch.from_numpy(np.array(img_resized)).float()
                        # Normalize the image by scaling pixel values to [-1, 1]
                        img_normalized = img_tensor / 127.5 - 1
                        img_normalized = img_normalized.permute(2, 0, 1)

                    condition_value[i] = img_normalized
                condition_values.append(condition_value.clone())

            all_focals/=(self.original_width/self.exo_width)
            all_c /=(self.original_width/self.exo_width)

            ego_poses[:, :, 1:3] *= -1
            exo_poses[:, :, 1:3] *= -1

            rel_poses =  torch.matmul(torch.linalg.inv(ego_poses).unsqueeze(1), exo_poses)

            return {
                    'pixel_values': pixel_values,
                    "pixel_files":pixel_files, 
                    "condition_values":condition_values,
                    "condition_files":condition_files,
                    "ego_poses":ego_poses,
                    "exo_poses":exo_poses,
                    'scene': scene_folder,
                    'focal': all_focals,
                    'c': all_c,
                    'ego_images':ego_images,
                    'ego_files': pixel_files,
                    "idx":idx,
                    "rel_poses": rel_poses.squeeze(),
                    }


