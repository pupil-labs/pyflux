import os
import cv2
from pathlib import Path


def explode_recording(video_path, offset, delta, frames_path):
    if not os.path.isdir(frames_path):
        os.makedirs(frames_path)
    cap = cv2.VideoCapture(video_path)
    counter = 0
    success = True
    while success:
        success, frame = cap.read()
        counter += 1
        if counter % delta == 0 and counter > offset:
            filename = "frame_" + f"000000{counter//delta-offset//delta}"[-5:] + ".png"
            cv2.imwrite(str(frames_path / f"{filename}"), frame)
    cap.release()


if __name__ == "__main__":

    video_path = "/cluster/users/Kai/IMG_9698.MOV"
    offset = 0
    delta = 5
    frames_path = Path("/cluster/users/Kai/nerfstudio/data/kitchen_talk/raw_frames")

    explode_recording(video_path, offset, delta, frames_path)
