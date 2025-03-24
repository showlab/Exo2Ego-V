import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import warnings
import numpy as np
import pdb
import math
import shutil
import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.stream_id import StreamId
import json
import torch

def get_ego_intrinsic(scene,width=256, height=256):
    vrs_file = '"Your Data Folder"/takes/{}'.format(scene)
    if not os.path.exists(vrs_file):
        return None
    for name in os.listdir(vrs_file):
        if (("aria" in name or "Aria" in name) and name.endswith(".vrs")):
            vrs_file = os.path.join(vrs_file,name)
            break
    if not os.path.isfile(vrs_file):
        return None

    vrs_data_provider = data_provider.create_vrs_data_provider(vrs_file)
    rgb_stream_id = StreamId("214-1")
    rgb_stream_label = vrs_data_provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = vrs_data_provider.get_device_calibration()
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
    X, Y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    proj = np.stack((X, Y), axis=-1)
    unproj = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            unproj[i, j, :] = rgb_camera_calibration.unproject(proj[i, j, :].reshape((2, 1))*(rgb_camera_calibration.get_image_size()[-1]/width))
    unproj = torch.from_numpy(unproj).to(torch.float32)
    unproj[:, :, 1:3] *= -1
    unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1)
    return unproj

unproj={}
path = '"Your Data Folder"/category'
ego_path = '"Your Output Folder of Ego Poses"/unproj_ego.pt'

for scene in os.listdir(path):
    value = get_ego_intrinsic(scene,32,32)
    if value !=None:
        unproj[scene] = value

torch.save(unproj, ego_path)
