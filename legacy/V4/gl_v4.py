import glfw
import numpy as np
import pyrr
from OpenGL.GL import *
from pyrr import Vector3, vector
from PIL import Image
import cv2

from pyflux.camera import Camera
from pyflux.mesh import HeatTriMesh, load_glb
from pyflux.pose_visualizer import PoseVisualizer
from pyflux.shader import DepthTextureShader, HeatMapShader, ShadowMapper, CircleShader
from pyflux.window import GLContext, GLFWWindow

record_flag = False
record_resolution = 800, 800
record_path = "/home/kd/Desktop/flux.mp4"

###########################################################################

meshes = load_glb("/cluster/users/Kai/nerfstudio/models/cubes.glb", subdivisions=0)

###########################################################################

global_cam = Camera()
global_cam.camera_pos = Vector3([0, 20, 35])
global_cam.camera_front = vector.normalise(-global_cam.camera_pos)
global_view = global_cam.get_view_matrix()

###########################################################################

pov_cam = Camera()

###########################################################################

width, height = 1200, 1200
lastX, lastY = width / 2, height / 2
first_mouse = True
left, right, forward, backward = False, False, False, False

###########################################################################

near_pane, far_plane = 0.01, 30.0

###########################################################################

window = GLFWWindow(width, height)
glfw.swap_interval(1)

###########################################################################

context = GLContext(FSAA_MODE=5)

###########################################################################

shadow_mapper = ShadowMapper(width=width, height=height)
heatmap_shader = HeatMapShader(cm="jet")
depth_texture_shader = DepthTextureShader(llc=0.5)
pose_visualizer = PoseVisualizer(z_depth=0.03, color=[0.0, 1.0, 0.0, 0.8])
circle_shader = CircleShader(radius=0.02, center=[0.75, 0.75])

###########################################################################

heat_tri_meshes = {key: HeatTriMesh(meshes[key]) for key in meshes.keys()}

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
    if key == glfw.KEY_R and action == glfw.PRESS:
        for key in meshes.keys():
            heat_tri_meshes[key].reset_heatmap()
    if key == glfw.KEY_P and action == glfw.PRESS:
        pose_visualizer._reset_poses()
    if key == glfw.KEY_G and action == glfw.PRESS:
        for key in meshes.keys():
            heat_tri_meshes[key].get_heatmap_to_GPU()
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

while not glfw.window_should_close(window.window):

    start = glfw.get_time()

    glfw.poll_events()

    #####################################################################

    update_global_cam()
    global_view = global_cam.get_view_matrix()

    update_pov_cam()
    pov_view = pov_cam.get_view_matrix()

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
    depth_texture_shader._set_uniforms(near_pane, far_plane)
    depth_texture_shader.draw(shadow_mapper.depth_map)

    circle_shader.draw()

    heatmap_shader.use()
    glBindTexture(GL_TEXTURE_2D, shadow_mapper.depth_map)
    heatmap_shader._set_uniforms(model, pov_view, global_view, projection)
    for key, mesh in heat_tri_meshes.items():
        mesh.draw_gl(heatmap_shader.shader)

    pose_visualizer.use()
    pose_visualizer.add_pose(np.linalg.inv(pov_view))
    pose_visualizer.set_uniforms(model, global_view, projection)
    pose_visualizer.draw()

    if record_flag:

        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGBA", (width, height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image = image.resize(record_resolution)
        out.write(np.array(image)[:, :, [2, 1, 0]])

    glfw.swap_buffers(window.window)

    end = glfw.get_time()

    print(1.0 / (end - start))

if record_flag:

    out.release()

glfw.terminate()
