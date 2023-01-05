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

import os

os.environ["__GL_FSAA_MODE"] = "11"

path = Path("/cluster/users/Kai/git/attention_flux_analysis")

global pos

###########################################################################

window = GLFWWindow(1200, 1200)


def key_input_clb(window, key, scancode, action, mode):
    global pos
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
    if key == glfw.KEY_L and action == glfw.REPEAT:
        pos = (
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, -0.5])) @ pos
        )
    if key == glfw.KEY_K and (action == glfw.REPEAT or action == glfw.PRESS):
        pos = (
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, +0.5])) @ pos
        )
    if key == glfw.KEY_L and (action == glfw.REPEAT or action == glfw.PRESS):
        pos = (
            pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, -0.5])) @ pos
        )
    if key == glfw.KEY_R and action == glfw.PRESS:
        heat = np.zeros(3 * n_triangles, dtype=np.float32)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, heat.nbytes, heat)


glfw.set_key_callback(window.window, key_input_clb)

###########################################################################


# with open(path / "versions" / "V2" / "vs.txt", "r") as ifile:
#     vertex_src = ifile.read()

# with open(path / "versions" / "V2" / "fs.txt", "r") as ifile:
#     fragment_src = ifile.read()

# with open(path / "versions" / "V2" / "gs.txt", "r") as ifile:
#     geometry_src = str(ifile.read())
#     geometry_src = geometry_src.replace(
#         "##colormapcode##", generate_colormap_code("jet", 1000)
#     )

# shader = compileProgram(
#     compileShader(vertex_src, GL_VERTEX_SHADER),
#     compileShader(geometry_src, GL_GEOMETRY_SHADER),
#     compileShader(fragment_src, GL_FRAGMENT_SHADER),
# )

# eye_shader = get_eye_shader()

eye = GLEye()

###########################################################################

# n_triangles, buffer = load_obj(path / "meshes" / "blob.obj")

# ###########################################################################

# VAO = glGenVertexArrays(1)
# VBO = glGenBuffers(1)

# glBindVertexArray(VAO)

# glBindBuffer(GL_ARRAY_BUFFER, VBO)
# glBufferData(GL_ARRAY_BUFFER, buffer.nbytes, buffer, GL_STATIC_DRAW)
# glEnableVertexAttribArray(0)
# glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, buffer.itemsize * 9, ctypes.c_void_p(0))
# glEnableVertexAttribArray(1)
# glVertexAttribPointer(
#     1, 2, GL_FLOAT, GL_FALSE, buffer.itemsize * 9, ctypes.c_void_p(12)
# )
# glEnableVertexAttribArray(2)
# glVertexAttribPointer(
#     2, 3, GL_FLOAT, GL_FALSE, buffer.itemsize * 9, ctypes.c_void_p(20)
# )
# glEnableVertexAttribArray(3)
# glVertexAttribPointer(
#     3, 1, GL_FLOAT, GL_FALSE, buffer.itemsize * 9, ctypes.c_void_p(32)
# )

# glBindVertexArray(0)

################

# eye_vertices = np.load(path / "meshes" / "eyeball.npy")
# eye_vertices = np.asarray(eye_vertices, dtype=np.float32)
# eye_buffer = np.ravel(eye_vertices)

# VAOeye = glGenVertexArrays(1)
# VBOeye = glGenBuffers(1)

# glBindVertexArray(VAOeye)

# glBindBuffer(GL_ARRAY_BUFFER, VBOeye)
# glBufferData(GL_ARRAY_BUFFER, eye_buffer.nbytes, eye_buffer, GL_STATIC_DRAW)
# glEnableVertexAttribArray(0)
# glVertexAttribPointer(
#     0, 3, GL_FLOAT, GL_FALSE, eye_buffer.itemsize * 12, ctypes.c_void_p(0)
# )
# glEnableVertexAttribArray(1)
# glVertexAttribPointer(
#     1, 2, GL_FLOAT, GL_FALSE, eye_buffer.itemsize * 12, ctypes.c_void_p(12)
# )
# glEnableVertexAttribArray(2)
# glVertexAttribPointer(
#     2, 3, GL_FLOAT, GL_FALSE, eye_buffer.itemsize * 12, ctypes.c_void_p(20)
# )
# glEnableVertexAttribArray(3)
# glVertexAttribPointer(
#     3, 4, GL_FLOAT, GL_FALSE, eye_buffer.itemsize * 12, ctypes.c_void_p(32)
# )

# glBindVertexArray(0)

###########################################################################

# heat = np.zeros(n_triangles * 3, dtype=np.float32)
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


# projection = pyrr.matrix44.create_perspective_projection_matrix(45, 1 / 1, 0.1, 100)
# pos = pyrr.matrix44.create_from_translation([0.0, 0.0, -3.0])
# view = pyrr.matrix44.create_look_at([0, 0, 0], [0, 0, -1], [0, 1, 0])
# eyepos = pyrr.matrix44.create_from_translation([0.0, 0.0, -100.0])

# glUseProgram(shader)

# model_loc = glGetUniformLocation(shader, "model")

# projection_loc = glGetUniformLocation(shader, "projection")
# glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)

# view_loc = glGetUniformLocation(shader, "view")
# glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

# objectColor_loc = glGetUniformLocation(shader, "objectColor")
# glUniform3f(objectColor_loc, 0.7, 0.7, 0.7)

# lightColor_loc = glGetUniformLocation(shader, "lightColor")
# glUniform3f(lightColor_loc, 1.0, 1.0, 1.0)

# lightPos_loc = glGetUniformLocation(shader, "lightPos")
# glUniform3f(lightPos_loc, 1.0, 1.0, 1.0)

# viewPos_loc = glGetUniformLocation(shader, "viewPos")
# glUniform3f(viewPos_loc, *view[3, :3])


while not glfw.window_should_close(window.window):

    glfw.poll_events()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # glUseProgram(shader)
    # glBindVertexArray(VAO)
    # rot_y = pyrr.Matrix44.from_y_rotation(2.6 * glfw.get_time())
    # rot_x = pyrr.Matrix44.from_x_rotation(1.55 * glfw.get_time())
    # rot_z = pyrr.Matrix44.from_z_rotation(0.3 * glfw.get_time())
    # model = rot_z @ rot_x @ rot_y @ pos
    # glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    # glDrawArrays(GL_TRIANGLES, 0, n_triangles * 3)
    # glBindVertexArray(0)

    eye.draw_gl()

    glfw.swap_buffers(window.window)

glfw.terminate()
