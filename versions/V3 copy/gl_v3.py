import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from pyflux.window import GLFWWindow
from pyflux.mesh import load_obj
import numpy as np
from pyflux.colormap import generate_colormap_glsl_code
from pathlib import Path
from pyflux.gleye import GLEye
import time
import trimesh
from PIL import Image
from pyflux.camera import Camera
import os

os.environ["__GL_FSAA_MODE"] = "5"

path = Path("/cluster/users/Kai/git/attention_flux_analysis")

global pos

cam = Camera()
WIDTH, HEIGHT = 600, 600
lastX, lastY = WIDTH / 2, HEIGHT / 2
first_mouse = True
left, right, forward, backward = False, False, False, False


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
    # if key in [glfw.KEY_W, glfw.KEY_S, glfw.KEY_D, glfw.KEY_A] and action == glfw.RELEASE:
    #     left, right, forward, backward = False, False, False, False


# do the movement, call this function in the main loop
def do_movement():
    if left:
        cam.process_keyboard("LEFT", 0.1)
    if right:
        cam.process_keyboard("RIGHT", 0.1)
    if forward:
        cam.process_keyboard("FORWARD", 0.1)
    if backward:
        cam.process_keyboard("BACKWARD", 0.1)


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

    cam.process_mouse_movement(xoffset, yoffset)


mesh = trimesh.load("/cluster/users/Kai/nerfstudio/models/cubes.glb")
objects = {}
for k, g in mesh.geometry.items():
    for _ in range(2):
        mesh.geometry[k] = mesh.geometry[k].subdivide()
    idx = np.ravel(mesh.geometry[k].faces)
    v = mesh.geometry[k].vertices[idx]
    vn = mesh.geometry[k].vertex_normals[idx]
    t = mesh.geometry[k].visual.uv[idx]
    color = np.ones_like(vn) * 0.7
    temp = np.hstack((v, t, vn, color))
    objects[k] = np.asarray(temp, dtype=np.float32)

window = GLFWWindow(1200, 1200)

# set the mouse position callback
glfw.set_cursor_pos_callback(window.window, mouse_look_clb)
# set the keyboard input callback
glfw.set_key_callback(window.window, key_input_clb)
# capture the mouse cursor
glfw.set_input_mode(window.window, glfw.CURSOR, glfw.CURSOR_DISABLED)

###########################################################################


with open(path / "versions" / "V3" / "vs.txt", "r") as ifile:
    vertex_src = ifile.read()

with open(path / "versions" / "V3" / "fs.txt", "r") as ifile:
    fragment_src = ifile.read()

shader = compileProgram(
    compileShader(vertex_src, GL_VERTEX_SHADER),
    compileShader(fragment_src, GL_FRAGMENT_SHADER),
)

###########################################################################

n_objects = len(objects.keys())
n_triangles = [len(objects[k]) // 3 for k in objects.keys()]
# images = [
#     np.array(
#         Image.open(
#             "/cluster/users/Kai/git/attention_flux_analysis/versions/V3/concrete.jpg"
#         )
#         .convert("RGBA")
#         .resize((800, 600))
#         .getdata(),
#         np.uint8,
#     )
#     for i in range(n_objects)
# ]

VAO = glGenVertexArrays(n_objects)
VBO = glGenBuffers(n_objects)
# TEX = glGenTextures(n_objects)

SSBO = glGenBuffers(n_objects)

for i in range(n_objects):

    k = list(objects.keys())[i]

    buffer = np.ravel(objects[k])

    glBindVertexArray(VAO[i])

    glBindBuffer(GL_ARRAY_BUFFER, VBO[i])
    glBufferData(GL_ARRAY_BUFFER, buffer.nbytes, buffer, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(
        0, 3, GL_FLOAT, GL_FALSE, buffer.itemsize * 11, ctypes.c_void_p(0)
    )
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(
        1, 2, GL_FLOAT, GL_FALSE, buffer.itemsize * 11, ctypes.c_void_p(12)
    )
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(
        2, 3, GL_FLOAT, GL_FALSE, buffer.itemsize * 11, ctypes.c_void_p(20)
    )
    glEnableVertexAttribArray(3)
    glVertexAttribPointer(
        3, 3, GL_FLOAT, GL_FALSE, buffer.itemsize * 11, ctypes.c_void_p(32)
    )

    heat = np.zeros(n_triangles, dtype=np.float32)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO[i])
    glBufferData(GL_SHADER_STORAGE_BUFFER, heat.nbytes, heat, GL_DYNAMIC_COPY)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, SSBO)

    # glBindTexture(GL_TEXTURE_2D, TEX[i])
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # glTexImage2D(
    #     GL_TEXTURE_2D,
    #     0,
    #     GL_RGBA,
    #     800,
    #     600,
    #     0,
    #     GL_RGBA,
    #     GL_UNSIGNED_BYTE,
    #     images[i],
    # )

    # glBindVertexArray(0)

###########################################################################

# heat = np.zeros(n_triangles, dtype=np.float32)
# SSBO = glGenBuffers(1)
# glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO)
# glBufferData(GL_SHADER_STORAGE_BUFFER, heat.nbytes, heat, GL_DYNAMIC_COPY)
# glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, SSBO)

###########################################################################

glClearColor(0.0, 0.0, 0.0, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

###########################################################################

glUseProgram(shader)

projection = pyrr.matrix44.create_perspective_projection_matrix(45, 1 / 1, 0.1, 100)
pos = pyrr.matrix44.create_from_translation([0.0, 0.0, 0.0])

model_loc = glGetUniformLocation(shader, "model")
view_loc = glGetUniformLocation(shader, "view")
viewPos_loc = glGetUniformLocation(shader, "viewPos")

projection_loc = glGetUniformLocation(shader, "projection")
glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)

lightColor_loc = glGetUniformLocation(shader, "lightColor")
glUniform3f(lightColor_loc, 1.0, 1.0, 1.0)
lightPos_loc = glGetUniformLocation(shader, "lightPos")
glUniform3f(lightPos_loc, 30.0, 30.0, 30.0)
#######################################################################

glUseProgram(shader)

while not glfw.window_should_close(window.window):

    glfw.poll_events()

    do_movement()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    rot_z = pyrr.Matrix44.from_z_rotation(0.7 * glfw.get_time())
    model = rot_z @ pos

    for i in range(len(objects)):

        glBindVertexArray(VAO[i])

        glUniformMatrix4fv(model_loc, 1, GL_FALSE, pos)

        view = cam.get_view_matrix()
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniform3f(viewPos_loc, *view[3, :3])

        glDrawArrays(GL_TRIANGLES, 0, n_triangles[i] * 3)

    glfw.swap_buffers(window.window)

glfw.terminate()
