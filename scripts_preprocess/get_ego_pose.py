
import numpy as np
import pandas as pd
import pdb
import json
from projectaria_tools.core import mps
import os
import cv2

base_pose_path  = "Your Output Folder"
final_path = "Your Output Folder of the Frame Extraction Process"
take_path = '"Your Data Folder"/takes.json'

def find_video_directory(root_directory, target_filename):
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(target_filename):
                return file

    return None

def construct_extrinsic_matrix(rotation_matrix, translation_vector):
    # [R | t]
    extrinsic_matrix = np.vstack((np.hstack((rotation_matrix, translation_vector.reshape(-1, 1))),
                                np.array([0, 0, 0, 1])))
    return extrinsic_matrix

def get_aria_start_end_sec( timesync_path, start_idx, end_idx):
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

def get_ego_pose(scene_folder):
    segments = {}
    with open(take_path) as json_file: 
        take_metadata = json.load(json_file)
        for meta in take_metadata:
            if scene_folder in meta['take_name']:
                segments[meta['take_name']] =  [meta['timesync_start_idx'], meta['timesync_end_idx']]
    for scene,time_list in segments.items():

        video_path = os.path.join(final_path,scene, "ego")
        if not os.path.exists(video_path):
            continue
        ss = scene.rsplit('_', 1)[0]
        start_idx, end_idx= time_list[0], time_list[1]
        start_sec, end_sec = get_aria_start_end_sec('"Your Data Folder"/captures/{}/timesync.csv'.format(ss), start_idx, end_idx)
        closed_loop_trajectory_filepath = '"Your Data Folder"/takes/{}/trajectory/closed_loop_trajectory.csv'.format(scene)
        mps_trajectory = mps.read_closed_loop_trajectory(closed_loop_trajectory_filepath)
        if mps_trajectory==[]:
            break
        rotations = []
        translations = []
        poses = []
        for pose in mps_trajectory:
            if start_sec and pose.tracking_timestamp.total_seconds() < start_sec:
                continue
            if end_sec and pose.tracking_timestamp.total_seconds() >= end_sec:
                break
            tracking_timestamp_ns = int(pose.tracking_timestamp.total_seconds()*1e9)
            T_world_device = pose.transform_world_device
            rotation = T_world_device.rotation().to_matrix()
            translation = T_world_device.translation()
            poses.append(construct_extrinsic_matrix(rotation,translation))

        target_filename = "214-1.mp4" 
        base_folder=os.path.join('"Your Data Folder"/takes"',scene,"frame_aligned_videos","downscaled","448")
        video_path = os.path.join(base_folder,find_video_directory(base_folder,target_filename))
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames ==0:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames==0:
                pdb.set_trace()
        per_pose =float(len(poses)/total_frames)
        video_path = os.path.join(final_path, scene, "ego")
        actions = os.listdir(video_path)
        pose_key={}
        pose_dir = os.path.join(base_pose_path, scene)
        if not os.path.exists(pose_dir):
            os.makedirs(pose_dir)
        for action in actions:
            action_path = os.path.join(video_path, action)
            images = os.listdir(action_path)
            for image in images:
                index = int(int(image.split('.')[0].split('_')[-1])*per_pose)
                pose =  poses[index]
                if image not in  pose_key:
                    pose_key[image] = pose.tolist()
        with open(os.path.join(pose_dir, "pose.json"), "w") as json_file:
            json.dump(pose_key, json_file, indent=4)
        print(scene)

names = []
for name in os.listdir(final_path):
    if os.path.exists(os.path.join(base_pose_path,name)) and len(os.listdir(os.path.join(base_pose_path,name)))>0:
        continue
    names.append(name)

for name in names:
    get_ego_pose(name)
