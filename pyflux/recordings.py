import glob
import json
from pathlib import Path
import numpy as np

import pandas as pd


def get_world_timestamps(recording_path):
    return pd.read_csv(recording_path / "world_timestamps.csv")["timestamp [ns]"].values


def get_relative_timestamps(recording_path):
    start_time = json.load(open(recording_path / "info.json", "r"))["start_time"]
    world_timestamps = get_world_timestamps(recording_path)
    relative_timestamps = (world_timestamps - start_time) / 1e9
    return relative_timestamps


def get_start_time(recording_path):
    return json.load(open(recording_path / "info.json", "r"))["start_time"]


def get_recording_ids_in_path(path):
    recording_ids = []
    for entry in glob.glob(str(path / "*")):
        recording_path = Path(entry)
        if recording_path.is_dir():
            try:
                get_start_time(recording_path)
                recording_ids.append(recording_path.name)
            except:
                print(f"Not appropriate recording, {recording_path}")
    return recording_ids


def get_world_camera_intrinsics_from_transforms(path):
    with open(path / "transforms.json", "r") as ifile:

        intrinsics_json = json.load(ifile)

        camera_matrix = np.eye(3)
        camera_matrix[0][0] = intrinsics_json["fl_x"]
        camera_matrix[1][1] = intrinsics_json["fl_y"]
        camera_matrix[0][2] = intrinsics_json["cx"]
        camera_matrix[1][2] = intrinsics_json["cy"]

        dist_coeffs = np.asarray([intrinsics_json["k1"],
                                  intrinsics_json["k2"],
                                  intrinsics_json["p1"],
                                  intrinsics_json["p2"]])

    return camera_matrix, dist_coeffs


def get_world_camera_intrinsics(path):
    with open(path / "scene_camera.json", "r") as ifile:
        intrinsics_json = json.load(ifile)
        camera_matrix = np.asarray(intrinsics_json["camera_matrix"])
        dist_coeffs = np.asarray(intrinsics_json["dist_coefs"])
    return camera_matrix, dist_coeffs
