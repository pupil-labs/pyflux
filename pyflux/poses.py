import json
import pickle

import cv2
import numpy as np
import pandas as pd
import scipy.stats
import torch

from pyflux.recordings import (
    get_start_time,
    get_world_camera_intrinsics_from_transforms,
    get_world_camera_intrinsics,
)


def convert_rim_pose_to_colmap_pose(
    pose, R=np.eye(3), slopes=np.ones(3), intercepts=np.zeros(3)
):

    pose = np.asarray(pose)

    colmap_pose = np.eye(4)
    colmap_pose[:3, :3] = R @ pose[:3, :3]
    colmap_pose[:3, 3] = slopes * (R @ pose[:3, 3]) + intercepts

    return colmap_pose


def convert_rim_pose_to_nerfstudio_convention(pose):

    rvec = np.asarray([pose["rotation_x"], pose["rotation_y"], pose["rotation_z"]])
    tvec = np.asarray(
        [pose["translation_x"], pose["translation_y"], pose["translation_z"]]
    )

    cloud_pose = np.eye(4)
    cloud_pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    cloud_pose[:3, 3] = tvec

    # This comes from the nerfstudio code
    nerfstudio_pose = cloud_pose.copy()
    nerfstudio_pose[0:3, 1:3] *= -1
    nerfstudio_pose = nerfstudio_pose[np.array([1, 0, 2, 3]), :]
    nerfstudio_pose[2, :] *= -1
    nerfstudio_pose = [[nerfstudio_pose[i][j] for j in range(4)] for i in range(4)]

    return nerfstudio_pose


def convert_nerfstudio_convention_to_ply_convention(nerfstudio_poses):
    py_poses = []
    for pose in nerfstudio_poses:
        pose = np.vstack((pose, np.asarray([0.0, 0.0, 0.0, 1.0])))
        converted_pose = pose.copy()
        converted_pose[1, :] = -pose[2, :].copy()
        converted_pose[2, :] = pose[1, :].copy()
        py_poses.append(converted_pose)
    return py_poses


def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return (
        torch.eye(3)
        + skew_sym_mat
        + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s ** 2 + 1e-8))
    )


def get_colmap_normalization(path):

    poses = torch.FloatTensor(get_poses_from_transforms_json_as_array(path))

    translation = poses[..., :3, 3]
    mean_translation = torch.mean(translation, dim=0)
    translation = mean_translation

    up = torch.mean(poses[:, :3, 1], dim=0)
    up = up / torch.linalg.norm(up)

    rotation = rotation_matrix(up, torch.Tensor([0, 0, 1]))

    transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)

    poses_transformed = transform @ poses
    scale_factor = 1.0 / torch.max(torch.abs(poses_transformed[:, :3, 3]))

    return transform, scale_factor


def get_poses_from_transforms_json_as_dict(path):
    transforms = json.load(open(path, "r"))
    poses = {
        frame["file_path"]: frame["transform_matrix"] for frame in transforms["frames"]
    }
    return poses


def get_pose_df_from_transforms_json(path):
    poses = get_poses_from_transforms_json_as_dict(path)
    df = pd.DataFrame({"pose": [item for key,item in poses.items()], "key": [key for key, item in poses.items()]})
    df["gaze_x"]=0.
    df["gaze_y"]=0.
    df["gaze_z"]=1.
    df = df.sort_values("key")
    df = df.reset_index(drop=True)    
    return df


def get_poses_from_transforms_json_as_array(path):
    transforms = json.load(open(path, "r"))
    poses = np.asarray([frame["transform_matrix"] for frame in transforms["frames"]])
    return poses


def calculate_rim_to_colmap_conversion(transforms_path_rim, transforms_path_colmap):

    transforms_rim = json.load(open(transforms_path_rim, "r"))
    transforms_colmap = json.load(open(transforms_path_colmap, "r"))

    transforms_rim_per_file = {
        entry["file_path"]: entry["transform_matrix"]
        for entry in transforms_rim["frames"]
    }
    transforms_colmap_per_file = {
        entry["file_path"]: entry["transform_matrix"]
        for entry in transforms_colmap["frames"]
    }

    def get_rot(transform):
        return np.asarray(transform)[:3, :3]

    def get_tvec(transform):
        return np.asarray(transform)[:3, 3]

    tvecs_cloud = np.asarray(
        [get_tvec(item) for key, item in transforms_rim_per_file.items()]
    )
    tvec_colmap = np.asarray(
        [get_tvec(item) for key, item in transforms_colmap_per_file.items()]
    )

    rots_cloud = []
    rots_cloud_inv = []
    rots_colmap = []
    tvecs_cloud = []
    tvecs_colmap = []

    for key, item in transforms_colmap_per_file.items():
        if key in transforms_rim_per_file.keys():

            rots_colmap.append(get_rot(transforms_colmap_per_file[key]))
            rots_cloud.append(get_rot(transforms_rim_per_file[key]))
            rots_cloud_inv.append(np.linalg.inv(rots_cloud[-1]))
            tvecs_colmap.append(get_tvec(transforms_colmap_per_file[key]))
            tvecs_cloud.append(get_tvec(transforms_rim_per_file[key]))

    rots_cloud = np.asarray(rots_cloud)
    rots_cloud_inv = np.asarray(rots_cloud_inv)
    rots_colmap = np.asarray(rots_colmap)
    tvecs_cloud = np.asarray(tvecs_cloud)
    tvecs_colmap = np.asarray(tvecs_colmap)

    R = np.median(np.einsum("ikj,ijl->ikl", rots_colmap, rots_cloud_inv), axis=0)

    res = (R @ tvecs_cloud.T).T

    slopes = np.zeros(3)
    intercepts = np.zeros(3)
    for i in [0, 1, 2]:
        reg = scipy.stats.linregress(res[:, i], tvecs_colmap[:, i])
        slopes[i] = reg.slope
        intercepts[i] = reg.intercept

    return R, slopes, intercepts


class PoseConverter:
    def __init__(self, transforms_path_rim, transforms_path_colmap):

        self.transforms_path_rim = transforms_path_rim
        self.transforms_path_colmap = transforms_path_colmap
        self.R, self.slopes, self.intercepts = calculate_rim_to_colmap_conversion(
            self.transforms_path_rim, self.transforms_path_colmap
        )
        self.transform, self.scale_factor = get_colmap_normalization(
            self.transforms_path_colmap
        )

    def convert_pose(self, pose):
        pose = np.asarray(
            [
                convert_rim_pose_to_colmap_pose(
                    pose, self.R, self.slopes, self.intercepts
                )
            ]
        )
        pose = torch.FloatTensor(pose)
        pose = self.transform @ pose
        pose[:, :3, 3] *= self.scale_factor
        pose = convert_nerfstudio_convention_to_ply_convention(pose)
        pose = np.asarray(pose, dtype=np.float32)[0]
        return pose


def add_3d_gaze_rays(df, camera_matrix, dist_coeffs):
    points = df[["gaze x [px]", "gaze y [px]"]].values
    rays_3d = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=None)
    n_points = rays_3d.shape[0]
    rays_3d = np.hstack((rays_3d[:, 0, :], np.ones((n_points, 1))))
    rays_3d /= np.linalg.norm(rays_3d, axis=1, keepdims=True)
    df["gaze_x"] = rays_3d[:, 0]
    df["gaze_y"] = rays_3d[:, 1]
    df["gaze_z"] = rays_3d[:, 2]
    return df


def get_gaze_and_pose_df(recording_path, transforms_path):

    start_time = get_start_time(recording_path)

    df = pd.read_csv(recording_path / "gaze.csv")
    df["relative_timestamp"] = (df["timestamp [ns]"] - start_time) / 1e9
    df["pose"] = [np.eye(4) for _ in range(len(df))]
    df["pose_indicator"] = [0 for _ in range(len(df))]

    camera_matrix, dist_coeffs = get_world_camera_intrinsics_from_transforms(
        transforms_path
    )
    df = add_3d_gaze_rays(df, camera_matrix, dist_coeffs)

    poses_raw = pickle.load(open(recording_path / "poses.p", "br"))

    poses = [convert_rim_pose_to_nerfstudio_convention(pose) for pose in poses_raw]
    start_timestamps = [pose["start_timestamp"] for pose in poses_raw]
    end_timestamps = [pose["end_timestamp"] for pose in poses_raw]

    df_poses = pd.DataFrame(
        {
            "pose": poses,
            "start_timestamp": start_timestamps,
            "end_timestamp": end_timestamps,
        }
    )

    t = df["relative_timestamp"].values
    ts = df_poses["start_timestamp"].values
    te = df_poses["end_timestamp"].values

    N = len(df)
    M = len(df_poses)
    i = 0
    j = 0
    while i < N:
        if t[i] < ts[j]:
            i += 1
        if ts[j] <= t[i] and t[i] < te[j]:
            df.at[i, "pose_indicator"] = 1
            df.at[i, "pose"] = np.asarray(df_poses["pose"][j])
            i += 1
        if t[i] >= te[j] and j + 1 < M:
            j += 1
        if t[i] >= te[j] and j + 1 == M:
            break

    return df


def get_pose_df(transforms_path):

    start_time = get_start_time(recording_path)

    df = pd.read_csv(recording_path / "gaze.csv")
    df["relative_timestamp"] = (df["timestamp [ns]"] - start_time) / 1e9
    df["pose"] = [np.eye(4) for _ in range(len(df))]
    df["pose_indicator"] = [0 for _ in range(len(df))]

    camera_matrix, dist_coeffs = get_world_camera_intrinsics_from_transforms(
        transforms_path
    )
    df = add_3d_gaze_rays(df, camera_matrix, dist_coeffs)

    poses_raw = pickle.load(open(recording_path / "poses.p", "br"))

    poses = [convert_rim_pose_to_nerfstudio_convention(pose) for pose in poses_raw]
    start_timestamps = [pose["start_timestamp"] for pose in poses_raw]
    end_timestamps = [pose["end_timestamp"] for pose in poses_raw]

    df_poses = pd.DataFrame(
        {
            "pose": poses,
            "start_timestamp": start_timestamps,
            "end_timestamp": end_timestamps,
        }
    )

    t = df["relative_timestamp"].values
    ts = df_poses["start_timestamp"].values
    te = df_poses["end_timestamp"].values

    N = len(df)
    M = len(df_poses)
    i = 0
    j = 0
    while i < N:
        if t[i] < ts[j]:
            i += 1
        if ts[j] <= t[i] and t[i] < te[j]:
            df.at[i, "pose_indicator"] = 1
            df.at[i, "pose"] = np.asarray(df_poses["pose"][j])
            i += 1
        if t[i] >= te[j] and j + 1 < M:
            j += 1
        if t[i] >= te[j] and j + 1 == M:
            break

    return df
