import glob
import json
import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests

API_KEY = "nEnyZQz9vHiDGG5dUpFgtuH3jm5p87B7mJjXKqHCTT8d"
API_URL = "https://api.cloud.pupil-labs.com/v2"

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def api_get(path):
    url = f"{API_URL}/{path}"
    if requests.get(url, headers={"api-key": API_KEY}).json()["status"] == "success":
        return requests.get(url, headers={"api-key": API_KEY}).json()["result"]
    else:
        error = requests.get(url, headers={"api-key": API_KEY}).json()["message"]
        log.error(error)
        raise (Exception(error))
    
def get_enrichment_dict(enrichment_id, project_id, workspace_id):
    log.info(f"fetching enrichment:{enrichment_id}")
    enrichment_dict = api_get(f"/workspaces/{workspace_id}/projects/{project_id}/enrichments/{enrichment_id}")       
    return enrichment_dict

def get_project_recordings(project_id, workspace_id):
    log.info(f"fetching recordings in project:{project_id}")
    project_recordings = api_get(f"/workspaces/{workspace_id}/projects/{project_id}/recordings")
    return project_recordings

def get_camera_poses_dict(enrichment_id, project_id, workspace_id):
    enrichment_dict = get_enrichment_dict(enrichment_id, project_id, workspace_id)
    assert enrichment_dict["kind"] == "slam-mapper"
    markerless_id = enrichment_dict["args"]["markerless_id"]
    
    project_recordings = get_project_recordings(project_id, workspace_id)
    
    camera_poses = {}
    for project_recording in project_recordings:
        recording_id = project_recording["recording_id"]
        log.info(f"fetching camera poses for recording:{recording_id}")
        temp = api_get(f"/workspaces/{workspace_id}/markerless/{markerless_id}/recordings/{recording_id}/camera_pose.json")
        if len(temp)>0:
            camera_poses[recording_id] = temp
    return camera_poses
    
def download_url(path, save_path, chunk_size=128):
    url = f"{API_URL}/{path}"
    r = requests.get(url, stream=True, headers={"api-key": API_KEY})
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def download_recording(recording_id, workspace_id, download_path):
    os.makedirs(download_path, exist_ok=True)
    download_url(f"/workspaces/{workspace_id}/recordings:raw-data-export?ids={recording_id}", download_path / f"{recording_id}.zip", chunk_size=128)
    shutil.unpack_archive(download_path / f"{recording_id}.zip", download_path / f"{recording_id}")
    os.remove(download_path / f"{recording_id}.zip")
    for file_source in glob.glob(str(download_path / f"{recording_id}/*/*")):
        file_source = Path(file_source)
        file_destination = file_source.parents[1] / file_source.name
        shutil.move(file_source,file_destination)

def download_raw_recording(recording_id, workspace_id, download_path):
    os.makedirs(download_path, exist_ok=True)
    download_url(f"/workspaces/{workspace_id}/recordings/{recording_id}.zip", download_path / f"{recording_id}.zip", chunk_size=128)
    shutil.unpack_archive(download_path / f"{recording_id}.zip", download_path / f"{recording_id}")
    os.remove(download_path / f"{recording_id}.zip")
    for file_source in glob.glob(str(download_path / f"{recording_id}/*/*")):
        file_source = Path(file_source)
        file_destination = file_source.parents[1] / file_source.name
        shutil.move(file_source,file_destination)