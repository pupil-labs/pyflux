import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
from cloud_api import *
from poses import convert_rim_pose_to_nerfstudio_convention
from recordings import get_relative_timestamps
import pickle

from paths import base_path

###########################################################

transforms = {
    "fl_x": 768.5348604948086,
    "fl_y": 768.2297161774204,
    "cx": 552.5247441952499,
    "cy": 539.0126768409235,
    "w": 1088,
    "h": 1080,
    "camera_model": "OPENCV",
    "k1": -0.29360766080176726,
    "k2": 0.07162415902612174,
    "p1": 0.0013999923091842345,
    "p2": 0.00039728879102343754,
    "frames": [],
}


def get_frames_with_poses(camera_poses, relative_timestamps):
    frames = []
    for pose in camera_poses:
        ts = pose["start_timestamp"]
        n_frame = np.argmin(np.abs(relative_timestamps - ts))
        frames.append([n_frame, convert_rim_pose_to_nerfstudio_convention(pose)])
    return frames


def get_localized_frames_dict(camera_poses_dict, download_path):
    localized_frames_dict = {}
    for RECORDING_ID in camera_poses_dict.keys():
        recording_path = Path(download_path / f"{RECORDING_ID}")
        relative_timestamps = get_relative_timestamps(recording_path)
        frames = get_frames_with_poses(
            camera_poses_dict[RECORDING_ID], relative_timestamps
        )
        localized_frames_dict[RECORDING_ID] = frames
    return localized_frames_dict


def write_raw_frames(localized_frames, image_path, n_per_recording=100):

    os.makedirs(image_path, exist_ok=True)

    counter = 0
    for RECORDING_ID in localized_frames.keys():
        path = Path(download_path / f"{RECORDING_ID}")
        videofile = glob.glob(str(path / "*.mp4"))[0]
        cap = cv2.VideoCapture(videofile)
        np.random.shuffle(localized_frames[RECORDING_ID])
        for frame_id, pose in localized_frames[RECORDING_ID][:n_per_recording]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            success, frame = cap.read()
            if success:
                counter += 1
                filename = f"frame_" + f"000000{counter}"[-5:] + ".png"
                transforms["frames"].append(
                    {"file_path": "images/" + str(filename), "transform_matrix": pose}
                )
                transforms["rec_id"] = RECORDING_ID
                cv2.imwrite(str(image_path / filename), frame)
        cap.release()

    json.dump(transforms, open(image_path.parent / "transforms_cloud.json", "w"))


if __name__ == "__main__":

    with open(os.path.join(Path(__file__).parent, "config.json"), "r") as f:
        config = json.load(f)
        WORKSPACE_ID = config["WORKSPACE_ID"]
        PROJECT_ID = config["PROJECT_ID"]
        ENRICHMENT_ID = config["ENRICHMENT_ID"]
        EXPERIMENT_NAME = config["EXPERIMENT_NAME"]

    download_path = base_path / "recordings" / f"{EXPERIMENT_NAME}"

    camera_poses_dict = get_camera_poses_dict(ENRICHMENT_ID, PROJECT_ID, WORKSPACE_ID)

    recording_ids_list = list(camera_poses_dict.keys())

    for RECORDING_ID in recording_ids_list:
        download_recording(RECORDING_ID, WORKSPACE_ID, download_path)
        pickle.dump(
            camera_poses_dict[RECORDING_ID],
            open(download_path / RECORDING_ID / "poses.p", "bw"),
        )

    localized_frames_dict = get_localized_frames_dict(camera_poses_dict, download_path)

    raw_frames_path = base_path / "data" / f"{EXPERIMENT_NAME}/raw_frames"

    write_raw_frames(localized_frames_dict, raw_frames_path, n_per_recording=100)
