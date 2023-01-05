import glfw
import numpy as np
import pyrr
from OpenGL.GL import *
from pyrr import Vector3, vector
from PIL import Image
import cv2
import json
import torch

from pyflux.camera import Camera
from pyflux.mesh import HeatTriMesh, load_glb, load_ply
from pyflux.pose_visualizer import PoseVisualizer
from pyflux.shader import (
    DepthTextureShader,
    ShadowMapper,
    CircleShader,
    TexturedHeatMapShader,
    NormalShader,
)
from pyflux.window import GLContext, GLFWWindow

record_flag = True
record_resolution = 1000, 1000
record_path = "/home/kd/Desktop/flux.mp4"

###########################################################################

# meshes = load_glb("/home/kd/Desktop/flux/livingroom3.glb", subdivisions=0)

meshes = load_ply("/home/kd/Desktop/flux/livingroom.ply", subdivisions=1)
###########################################################################

global_cam = Camera()

# global_cam.camera_pos = Vector3([0.31316081, 0.40071118, 3.49734914])
# global_cam.camera_front = Vector3([-0.03519388, -0.35787908, -0.93310447])
# global_cam.camera_up = Vector3([-0.01348853, 0.93376794, -0.3576248])
# global_cam.camera_right = Vector3([0.99928947, 0.0, -0.03769018])

global_cam.camera_pos = Vector3([0.0, -10.0, -10.0])
global_cam.camera_front = -global_cam.camera_pos
global_cam.camera_up = Vector3([0, 0, 1])
global_cam.camera_right = Vector3([1, 0, 0])

global_view = global_cam.get_view_matrix()

###########################################################################

pov_cam = Camera()
pov_cam.camera_pos = Vector3([0.0, 0.0, 0.0])

###########################################################################

width, height = 1200, 1200
lastX, lastY = width / 2, height / 2
first_mouse = True
left, right, forward, backward = False, False, False, False

###########################################################################

near_pane, far_plane = 0.01, 100.0

###########################################################################

window = GLFWWindow(width, height)
glfw.swap_interval(1)

###########################################################################

context = GLContext(FSAA_MODE=11)

###########################################################################

# shadow_mapper = ShadowMapper(width=width, height=height)
# heatmap_shader = TexturedHeatMapShader(cm="jet")
# depth_texture_shader = DepthTextureShader(left_lower_corner=0.5)
pose_visualizer = PoseVisualizer(z_depth=0.01, color=[0.0, 1.0, 0.0, 0.8])
pose_visualizer_2 = PoseVisualizer(z_depth=0.01, color=[1.0, 1.0, 1.0, 0.8])

# circle_shader = CircleShader(radius=0.02, center=[0.75, 0.75])
# normal_shader = NormalShader()

############################################################################


def get_poses_from_json(transforms):
    poses = np.asarray(
        [np.asarray(entry["transform_matrix"]).T for entry in transforms["frames"]]
    )
    return poses


transforms = json.load(
    open("/cluster/users/Kai/nerfstudio/data/kitchen_cloud_rim/transforms.json", "r")
)
oriented_poses = get_poses_from_json(transforms)
poses = np.asarray(oriented_poses, dtype=np.float32)
n_poses = len(poses)

for pose in poses:
    pose_visualizer.add_pose(pose)


transforms = json.load(
    open("/cluster/users/Kai/nerfstudio/data/kitchen_rim/transforms.json", "r")
)
oriented_poses2 = get_poses_from_json(transforms)
poses2 = np.asarray(oriented_poses2, dtype=np.float32)
n_poses2 = len(poses)


for pose in poses2:
    pose_visualizer_2.add_pose(pose)

# rel_pose = np.eye(4)  # poses[-10].copy()

###########################################################################

# heat_tri_meshes = {key: HeatTriMesh(meshes[key], rel_pose) for key in meshes.keys()}

###########################################################################

# poses = np.einsum("lk,ikj->ilj", np.linalg.inv(rel_pose), poses)

###########################################################################

# the keyboard input callback
def key_input_clb(window, key, scancode, action, mode):
    global left, right, forward, backward, mode_
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
    # if key == glfw.KEY_R and action == glfw.PRESS:
    #     for key in meshes.keys():
    #         heat_tri_meshes[key].reset_heatmap()
    if key == glfw.KEY_P and action == glfw.PRESS:
        pose_visualizer._reset_poses()
    # if key == glfw.KEY_G and action == glfw.PRESS:
    #     for key in meshes.keys():
    #         heat_tri_meshes[key].get_heatmap_to_GPU()
    if key == glfw.KEY_M and action == glfw.PRESS:
        if mode_ == "world":
            mode_ = "depth"
        else:
            mode_ = "world"


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
    45, width / height, near_pane, far_plane
)
model = pyrr.matrix44.create_from_translation([0.0, 0.0, 0.0])

if record_flag:
    fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
    out = cv2.VideoWriter(record_path, fourcc, 20.0, record_resolution)


glfw.swap_interval(2)

counter = 0

while not glfw.window_should_close(window.window):

    glfw.poll_events()

    #####################################################################

    # update_global_cam()
    time = 0.1 * glfw.get_time()
    rot = cv2.Rodrigues(time * np.asarray([0, 1, 0]))[0]
    rot4 = np.eye(4)
    trans4 = np.eye(4)
    # trans4[:3, 3] = np.asarray([0.0, 0.0, 0.0])
    rot4[:3, :3] = rot
    global_view = (
        np.linalg.inv(trans4.T) @ rot4 @ global_cam.get_view_matrix() @ trans4.T
    )

    # global_view = global_cam.get_view_matrix()

    # global_view = np.linalg.inv(poses[counter].T)
    # global_view = np.linalg.inv(poses[300].T)
    # counter += 1  # global_cam.get_view_matrix()
    # if counter == n_poses:
    #    counter = 0

    update_pov_cam()
    # pov_view = pov_cam.get_view_matrix()
    pov_view = np.linalg.inv(poses[counter].T)
    counter += 1
    if counter == n_poses:
        counter = 0

    #####################################################################

    # glViewport(0, 0, shadow_mapper.width, shadow_mapper.height)
    # glBindFramebuffer(GL_FRAMEBUFFER, shadow_mapper.frame_buffer)
    # glClear(GL_DEPTH_BUFFER_BIT)

    # shadow_mapper.use()
    # shadow_mapper._set_uniforms(
    #     model, (projection.T @ pov_view.T).T
    # )  # note: in glsl you end up with the transpose of this

    # for key, mesh in heat_tri_meshes.items():
    #     mesh.draw_gl(shadow_mapper.shader, ssbo_slot=0)

    # #####################################################################

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glViewport(0, 0, *window.framebuffer_size)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # depth_texture_shader.use()
    # depth_texture_shader._set_uniforms(near_pane, far_plane)
    # depth_texture_shader.draw(shadow_mapper.depth_map)
    # circle_shader.draw()

    # for shader in [heatmap_shader]:
    #     shader.use()
    #     glActiveTexture(GL_TEXTURE0 + 0)
    #     glBindTexture(GL_TEXTURE_2D, shadow_mapper.depth_map)
    #     glActiveTexture(GL_TEXTURE0 + 1)
    #     glBindTexture(GL_TEXTURE_2D, heatmap_shader.texture)
    #     shader._set_uniforms(model, pov_view, global_view, projection)
    #     for key, mesh in heat_tri_meshes.items():
    #         mesh.draw_gl(shader.shader)

    pose_visualizer.use()
    # pose_visualizer.add_pose(np.linalg.inv(pov_view))
    pose_visualizer.set_uniforms(model, global_view, projection)
    pose_visualizer.draw()

    pose_visualizer_2.use()
    # pose_visualizer.add_pose(np.linalg.inv(pov_view))
    pose_visualizer_2.set_uniforms(model, global_view, projection)
    pose_visualizer_2.draw()

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
