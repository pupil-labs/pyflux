import json
import shutil
import os
import copy
from pathlib import Path
from paths import base_path

if __name__ == "__main__":
    with open(os.path.join(Path(__file__).parent, "config.json"), "r") as f:
        config = json.load(f)
        EXPERIMENT_NAME = config["EXPERIMENT_NAME"]

    data_path = base_path / "data" / EXPERIMENT_NAME
    temp_path = data_path / "temp"
    os.makedirs(temp_path, exist_ok=True)

    original_transforms = json.load(open(data_path / "transforms_cloud.json", "r"))
    consolidated_transforms = copy.deepcopy(original_transforms)
    consolidated_transforms["frames"] = []

    counter = 1
    for frame in original_transforms["frames"]:
        file_path = data_path / "raw_frames" / frame["file_path"].split("/")[1]
        if file_path.exists():
            filename = f"frame_" + f"000000{counter}"[-5:] + ".png"
            consolidated_transforms["frames"].append(
                {
                    "file_path": "images/" + str(filename),
                    "transform_matrix": frame["transform_matrix"],
                }
            )
            shutil.copy(file_path, temp_path / filename)
            counter += 1

    json.dump(
        consolidated_transforms, open(data_path / "consolidated_transforms.json", "w")
    )

    shutil.rmtree(data_path / "raw_frames")
    shutil.move(temp_path, data_path / "raw_frames")
    os.remove(data_path / "transforms_cloud.json")
    shutil.move(
        data_path / "consolidated_transforms.json", data_path / "transforms_cloud.json"
    )
