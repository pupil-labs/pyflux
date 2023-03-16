import os
import sys
from datetime import datetime
from pathlib import Path
import time
import json
import torch

from paths import base_path, nerfstudio_path

sys.path.append(str(nerfstudio_path / "scripts"))
from exporter import ExportPoissonMesh
from nerfstudio.configs import base_config as cfg
from nerfstudio.configs.method_configs import method_configs
from process_data import ProcessImages
from train import main as train_nerf


def compute_mesh(
    experiment_name="livingroom",
    timestamp=None,
    process=True,
    train=True,
    export=True,
    bbox=5,
):

    ##################################################################

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    data_path = base_path / "data" / f"{experiment_name}"
    raw_frames_path = data_path / "raw_frames"
    export_path = base_path / "exports" / f"{experiment_name}"
    model_path = base_path / f"outputs/{experiment_name}/nerfacto/{timestamp}"
    config_path = model_path / "config.yml"

    ##################################################################

    os.chdir(base_path)

    ##################################################################

    if process:

        process_config = {
            "data": raw_frames_path,
            "output_dir": data_path,
            "gpu": True,
        }

        ProcessImages(**process_config).main()

    ##################################################################

    torch.cuda.empty_cache()

    ##################################################################

    if train:

        train_config = method_configs["nerfacto"]
        train_config.experiment_name = experiment_name
        train_config.data = data_path
        train_config.pipeline.model.predict_normals = True
        train_config.timestamp = timestamp
        train_config.max_num_iterations = 20000
        train_config.viewer.quit_on_train_completion = True

        train_nerf(train_config)

    ##################################################################

    torch.cuda.empty_cache()

    ##################################################################

    if export:
        export_config = {
            "load_config": config_path,
            "output_dir": export_path,
            "target_num_faces": 50000,
            "num_pixels_per_side": 2048,
            "normal_output_name": "normals",
            "num_points": 1000000,
            "remove_outliers": True,
            "use_bounding_box": True,
            "bounding_box_min": (-bbox, -bbox, -bbox),
            "bounding_box_max": (bbox, bbox, bbox),
        }

        ExportPoissonMesh(**export_config).main()

    return timestamp


if __name__ == "__main__":
    with open(os.path.join(Path(__file__).parent, "config.json"), "r") as f:
        config = json.load(f)
        EXPERIMENT_NAME = config["EXPERIMENT_NAME"]
        bbox = config["bbox"]
    kwargs = {
        "experiment_name": EXPERIMENT_NAME,
        "timestamp": "2023-03-10_133533",
        "process": False,
        "train": False,
        "export": True,
        "bbox": bbox,
    }

    compute_mesh(**kwargs)
