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

###########################################################


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
        recording_path = Path(download_path + f"/{RECORDING_ID}")
        relative_timestamps = get_relative_timestamps(recording_path)
        frames = get_frames_with_poses(
            camera_poses_dict[RECORDING_ID], relative_timestamps
        )
        localized_frames_dict[RECORDING_ID] = frames
    return localized_frames_dict


def write_raw_frames(localized_frames, image_path, n_per_recording=100):

    os.makedirs(image_path, exist_ok=True)

    transforms = json.load(open("/home/kd/Desktop/transforms.json", "r"))
    transforms["frames"] = []

    counter = 0
    for RECORDING_ID in localized_frames.keys():
        path = Path(download_path + f"/{RECORDING_ID}")
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
                cv2.imwrite(str(image_path / filename), frame)
        cap.release()

    json.dump(transforms, open(image_path.parent / "transforms_cloud.json", "w"))


if __name__ == "__main__":

    # Museum downstairs
    # WORKSPACE_ID = "78cddeee-772e-4e54-9963-1cc2f62825f9"
    # PROJECT_ID = "cdfde655-3c8a-45c5-b6e2-5d5754d7a4f0"
    # ENRICHMENT_ID = "89f43353-cbe6-4fd2-b2c5-00f8421ee392"

    # Museum schraege
    # WORKSPACE_ID = "78cddeee-772e-4e54-9963-1cc2f62825f9"
    # PROJECT_ID = "8a786732-8817-4f38-a9e2-b177b136327a"
    # ENRICHMENT_ID = "40d8d8c8-008b-4d25-a7e1-48a7bc479293"

    # Museum multi
    WORKSPACE_ID = "78cddeee-772e-4e54-9963-1cc2f62825f9"
    PROJECT_ID = "3bf7d748-b22f-49e0-8e7d-83dcb177b332"
    ENRICHMENT_ID = "68d640fb-63de-47c0-87a9-a3ce0ba36223"

    experiment = "supermarket"

    download_path = Path("/cluster/users/Kai/nerfstudio/recordings/{experiment}")

    camera_poses_dict = get_camera_poses_dict(ENRICHMENT_ID, PROJECT_ID, WORKSPACE_ID)

    recording_ids_list = list(camera_poses_dict.keys())

    for RECORDING_ID in recording_ids_list:
        download_recording(RECORDING_ID, WORKSPACE_ID, download_path)
        pickle.dump(
            camera_poses_dict[RECORDING_ID],
            open(download_path / RECORDING_ID / "poses.p", "bw"),
        )

    localized_frames_dict = get_localized_frames_dict(camera_poses_dict, download_path)

    raw_frames_path = Path("/cluster/users/Kai/nerfstudio/data/museum_multi/raw_frames")

    write_raw_frames(localized_frames_dict, raw_frames_path, 100)
