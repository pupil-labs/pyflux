from pathlib import Path

import cv2
import glfw
import numpy as np
import pyrr
from OpenGL.GL import *
from PIL import Image
from pyrr import Vector3, vector
import torch

from pyflux.paths import base_path
from pyflux.camera import Camera
from pyflux.mesh import HeatTriMesh, load_ply
from pyflux.pose_visualizer import PoseVisualizer, GazeVisualizer
from pyflux.poses import PoseConverter, get_gaze_and_pose_df, rotation_matrix
from pyflux.recordings import get_recording_ids_in_path
from pyflux.shader import (
    CircleShader,
    DepthTextureShader,
    NormalShader,
    ShadowMapper,
    HeatMapShader,
)
from pyflux.window import GLContext, GLFWWindow

###########################################################################

experiment_name = "hinterhof2"
ply_path = base_path / "models"
data_path = base_path / "data"
export_path = base_path / "exports"
recording_path = base_path / f"recordings/{experiment_name}"
available_recording_ids = get_recording_ids_in_path(recording_path)
recording_id = available_recording_ids[1]
print(recording_id)

###########################################################################

record_flag = False
if record_flag:
    record_resolution = 1200, 800
    record_path = "/home/kd/Desktop/berlin_office_flux.mp4"
    fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
    out = cv2.VideoWriter(record_path, fourcc, 30.0, record_resolution)

###########################################################################

meshes = load_ply(ply_path / f"{experiment_name}.ply", subdivisions=1)

###########################################################################

global_cam = Camera()
global_cam.camera_pos = Vector3([0.0, -0.5, -2.8])
global_cam.camera_front = -global_cam.camera_pos
global_cam.camera_up = Vector3([0, 0, 1])
global_cam.camera_right = Vector3([1, 0, 0])

###########################################################################

pov_cam = Camera()

###########################################################################

width, height = 1600, 1200
lastX, lastY = width / 2, height / 2
first_mouse = True
left, right, forward, backward = False, False, False, False

###########################################################################

near_plane, far_plane = 0.01, 4.0

###########################################################################

window = GLFWWindow(width, height)
glfw.swap_interval(1)

###########################################################################

context = GLContext(FSAA_MODE=11)

###########################################################################

shadow_mapper = ShadowMapper(width=width, height=height)
heatmap_shader = HeatMapShader(
    texfile=export_path / f"{experiment_name}/material_0.png", cm="jet"
)
pose_visualizer = PoseVisualizer(z_depth=0.005, color=[0.0, 1.0, 0.0, 1.0])
gaze_visualizer = GazeVisualizer(z_depth=1.0, color=[1.0, 1.0, 0.0, 1.0])

llc, urc = 0.5, 0.75
depth_texture_shader = DepthTextureShader(llc=llc, urc=urc)
circle_shader = CircleShader(
    radius=0.025,
    center=2 * [llc + 0.5 * (urc - llc)],
    color=[1.0, 0.0, 0.0, 1.0],
    ratio=height / width,
    linewidth=2.0,
)

# normal_shader = NormalShader()

############################################################################

pose_converter = PoseConverter(
    data_path / f"{experiment_name}/transforms_cloud.json",
    data_path / f"{experiment_name}/transforms.json",
)
df_gaze = get_gaze_and_pose_df(
    recording_path / recording_id, data_path / experiment_name
)
df_gaze = df_gaze[df_gaze["pose_indicator"] == 1]
df_gaze = df_gaze.reset_index()
n_poses = len(df_gaze)

###########################################################################

heat_tri_meshes = {key: HeatTriMesh(meshes[key]) for key in meshes.keys()}

###########################################################################

# the keyboard input callback
def key_input_clb(window, key, scancode, action, mode):

    global left, right, forward, backward

    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    if key == glfw.KEY_W and action == glfw.PRESS:
        forward = True
    elif key == glfw.KEY_W and action == glfw.RELEASE:
        forward = False

    if key == glfw.KEY_S and action == glfw.PRESS:
        backward = True
    elif key == glfw.KEY_S and action == glfw.RELEASE:
        backward = False

    if key == glfw.KEY_A and action == glfw.PRESS:
        left = True
    elif key == glfw.KEY_A and action == glfw.RELEASE:
        left = False

    if key == glfw.KEY_D and action == glfw.PRESS:
        right = True
    elif key == glfw.KEY_D and action == glfw.RELEASE:
        right = False

    if key == glfw.KEY_R and action == glfw.PRESS:
        for key in meshes.keys():
            heat_tri_meshes[key].reset_heatmap()

    if key == glfw.KEY_P and action == glfw.PRESS:
        pose_visualizer._reset_poses()

    if key == glfw.KEY_G and action == glfw.PRESS:
        for key in meshes.keys():
            heat_tri_meshes[key].get_heatmap_to_GPU()


# do the movement, call this function in the main loop
def update_pov_cam():
    if left:
        pov_cam.process_keyboard("LEFT", 0.05)
    if right:
        pov_cam.process_keyboard("RIGHT", 0.05)
    if forward:
        pov_cam.process_keyboard("FORWARD", 0.05)
    if backward:
        pov_cam.process_keyboard("BACKWARD", 0.05)


def update_global_cam(radius=3, frequency=0.1):
    time = glfw.get_time()
    global_cam.camera_pos = radius * Vector3(
        [
            np.cos(frequency * time + np.pi / 2),
            1,
            np.sin(frequency * time + np.pi / 2),
        ]
    )
    global_cam.camera_front = -vector.normalise(global_cam.camera_pos)


# the mouse position callback function
def mouse_look_clb(window, xpos, ypos):
    global first_mouse, lastX, lastY

    if first_mouse:
        lastX = xpos
        lastY = ypos
        first_mouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos

    lastX = xpos
    lastY = ypos

    pov_cam.process_mouse_movement(xoffset, yoffset)


# set the mouse position callback
glfw.set_cursor_pos_callback(window.window, mouse_look_clb)
# set the keyboard input callback
glfw.set_key_callback(window.window, key_input_clb)
# capture the mouse cursor
glfw.set_input_mode(window.window, glfw.CURSOR, glfw.CURSOR_DISABLED)

##########################################################################

projection = pyrr.matrix44.create_perspective_projection_matrix(
    45, width / height, near_plane, far_plane
)
model = pyrr.matrix44.create_from_translation([0.4, 0.0, 0.0])

counter = 0

while not glfw.window_should_close(window.window):

    glfw.poll_events()

    #####################################################################

    # update_global_cam()
    time = 5.0 + 0.5 * np.cos(0.4 * glfw.get_time())

    rot = cv2.Rodrigues(time * np.asarray([0, 1, 0]))[0]
    rot4 = np.eye(4)
    rot4[:3, :3] = rot

    trans4 = np.eye(4)
    trans4[:3, 3] = np.asarray([0.0, 0.0, 0.0])

    global_view = (
        np.linalg.inv(trans4.T) @ rot4 @ global_cam.get_view_matrix() @ trans4.T
    )

    # global_view = global_cam.get_view_matrix()
    # global_view = np.linalg.inv(poses[counter].T)
    # global_view = np.linalg.inv(poses[300].T)
    # counter += 1  # global_cam.get_view_matrix()
    # if counter == n_poses:
    #    counter = 0

    # update_pov_cam()
    # pov_view = pov_cam.get_view_matrix()
    pose = pose_converter.convert_pose(df_gaze["pose"][counter])
    gaze_3d = np.asarray(
        df_gaze.iloc[counter][["gaze_x", "gaze_y", "gaze_z"]].values, dtype=float
    )

    R = rotation_matrix(
        torch.FloatTensor([0.0, 0.0, 1.0]), torch.FloatTensor(gaze_3d)
    ).numpy()

    gaze_pose = pose.copy()
    gaze_pose[:3, :3] = gaze_pose[:3, :3] @ R
    gaze_pose = gaze_pose.T @ model

    pov_view = np.linalg.inv(gaze_pose)
    counter += 4
    if counter > n_poses - 1:
        counter = 0

    #####################################################################

    glViewport(0, 0, shadow_mapper.width, shadow_mapper.height)
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_mapper.frame_buffer)
    glClear(GL_DEPTH_BUFFER_BIT)

    shadow_mapper.use()
    shadow_mapper._set_uniforms(
        model, (projection.T @ pov_view.T).T
    )  # note: in glsl you end up with the transpose of this

    for key, mesh in heat_tri_meshes.items():
        mesh.draw_gl(shadow_mapper.shader, ssbo_slot=0)

    # #####################################################################

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glViewport(0, 0, *window.framebuffer_size)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    depth_texture_shader.use()
    depth_texture_shader._set_uniforms(near_plane, far_plane)
    depth_texture_shader.draw(shadow_mapper.depth_map)

    circle_shader.use()
    circle_shader.draw()

    heatmap_shader.use()
    glActiveTexture(GL_TEXTURE0 + 0)
    glBindTexture(GL_TEXTURE_2D, shadow_mapper.depth_map)
    glActiveTexture(GL_TEXTURE0 + 1)
    glBindTexture(GL_TEXTURE_2D, heatmap_shader.texture)
    heatmap_shader._set_uniforms(model, pov_view, global_view, projection)
    for key, mesh in heat_tri_meshes.items():
        mesh.draw_gl(heatmap_shader.shader)

    # normal_shader.use()
    # normal_shader._set_uniforms(model, global_view, projection)
    # for key, mesh in heat_tri_meshes.items():
    #     mesh.draw_gl(normal_shader.shader)

    pose_visualizer.use()
    pose_visualizer.add_pose(pose.T)
    pose_visualizer.set_uniforms(model, global_view, projection)
    pose_visualizer.draw(last_n=10000)

    gaze_visualizer.use()
    gaze_visualizer.add_pose(np.linalg.inv(pov_view) @ np.linalg.inv(model))
    gaze_visualizer.set_uniforms(model, global_view, projection)
    gaze_visualizer.draw(last_n=1)

    if record_flag:
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGBA", (width, height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image = image.resize(record_resolution)
        out.write(np.array(image)[:, :, [2, 1, 0]])

    glfw.swap_buffers(window.window)

if record_flag:

    out.release()

glfw.terminate()
