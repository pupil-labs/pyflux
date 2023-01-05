import os

import glfw
from OpenGL.GL import *


class GLContext:
    def __init__(self, FSAA_MODE=5):

        os.environ["__GL_FSAA_MODE"] = f"{FSAA_MODE}"

        glClearColor(0.2, 0.2, 0.2, 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


class GLFWWindow:
    def __init__(self, width=600, height=600, pos_x=0, pos_y=0, FSAA_MODE=5):

        # initiale glfw library
        if not glfw.init():
            raise Exception("glfw can not be initialized!")

        self._configure_context()

        # create window
        self.window = glfw.create_window(width, height, "My OpenGL window", None, None)

        # check if window was created
        if not self.window:
            glfw.terminate()
            raise Exception("glfw window can not be created!")

        # set window's position
        glfw.set_window_pos(self.window, pos_x, pos_y)

        # make the context current
        self._make_context_current()

    def _make_context_current(self):
        glfw.make_context_current(self.window)

    def _configure_context(self):
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        # last line commented since glLineWidht deprecated in 4.3
        glfw.window_hint(glfw.SAMPLES, 4)

    @property
    def framebuffer_size(self):
        return glfw.get_framebuffer_size(self.window)
