from pathlib import Path
import os, json


with open(os.path.join(Path(__file__).parent, "config.json"), "r") as f:
    config = json.load(f)
    nerfstudio_path = Path(config["NERFSTUDIO_PATH"])
    base_path = Path(config["BASE_PATH"])
