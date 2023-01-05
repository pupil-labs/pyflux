import sys, os
import logging
import requests
import pandas as pd
import numpy as np
import json

import nerfstudio
from nerfstudio.process_data.colmap_utils import read_images_binary

API_KEY = "nEnyZQz9vHiDGG5dUpFgtuH3jm5p87B7mJjXKqHCTT8d"
API_URL = "https://api.cloud.pupil-labs.com/v2"
WORKSPACE_ID = "3e8760bb-5f2a-4fd7-8cb4-3a5c3b3a055a"
PROJECT_ID = "702f1d48-7a8a-4ec5-8032-09dccb2b2089"
ENRICHMENT_ID = "6551cb9d-f8fd-4d79-8f30-d74ec69794d6"
RECORDING_ID = "521b315e-4bb0-40c4-a084-1aa3185a7fe8"

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


log.info("Authenticating...")
api_get(f"/auth/profile")

log.info(f"fetching enrichment:{ENRICHMENT_ID}")
enrichment = api_get(
    f"/workspaces/{WORKSPACE_ID}/projects/{PROJECT_ID}/enrichments/{ENRICHMENT_ID}"
)
assert enrichment["kind"] == "slam-mapper"

markerless_id = enrichment["args"]["markerless_id"]

  # Reading the Camera poses from RIM
log.info(f"fetching recordings in project:{PROJECT_ID}")
project_recordings = api_get(
    f"/workspaces/{WORKSPACE_ID}/projects/{PROJECT_ID}/recordings"
)
print(project_recordings)

# for project_recording in project_recordings:
#     recording_id = project_recording["recording_id"]
#     log.info(f"fetching camera poses for recording:{recording_id}")
recording_id = RECORDING_ID
camera_poses = api_get(
    f"/workspaces/{WORKSPACE_ID}/markerless/{markerless_id}/recordings/{recording_id}/camera_pose.json"
)
