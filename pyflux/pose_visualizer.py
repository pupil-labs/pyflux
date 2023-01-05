import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


class PoseVisualizer:

    MAX_POSES = 100000

    def __init__(
        self,
        field_of_view=45.0,
        aspect_ratio=1.0,
        z_depth=0.1,
        color=[1.0, 1.0, 1.0, 1.0],
    ):

        self.field_of_view = field_of_view
        self.aspect_ratio = aspect_ratio
        self.z_depth = z_depth
        self.color = color

        self.poses = np.tile(np.ravel(np.eye(4, dtype=np.float32)), self.MAX_POSES)
        self.count = 0
        self.poses_to_add = []

        self._setup_vertices()
        self._setup_vertex_arrays()
        self._setup_shader()
        self._get_uniforms()

    def _reset_poses(self):
        self.poses = np.tile(np.ravel(np.eye(4, dtype=np.float32)), self.MAX_POSES)
        self.count = 0
        glBindBuffer(GL_ARRAY_BUFFER, self.pose_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.poses.nbytes, self.poses, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _setup_vertices(self):

        delta_x = np.cos(self.field_of_view / 2)
        delta_y = delta_x / self.aspect_ratio

        self.vertices = np.asarray(
            [
                [0, 0, 0],
                [-delta_x, -delta_y, -1.0],
                [+delta_x, -delta_y, -1.0],
                [+delta_x, +delta_y, -1.0],
                [-delta_x, +delta_y, -1.0],
            ]
        )
        self.vertices *= self.z_depth
        self.vertices = np.ravel(self.vertices)
        self.vertices = np.asarray(self.vertices, dtype=np.float32)
        self.indices = np.asarray(
            [0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 2, 3, 3, 4, 4, 1], dtype=np.uint32
        )
        self.n_indices = len(self.indices)

    def _setup_vertex_arrays(self):

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        #######################################################3

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(
            GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW
        )

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0,
            3,
            GL_FLOAT,
            GL_FALSE,
            self.vertices.itemsize * 3,
            ctypes.c_void_p(0),
        )

        #######################################################

        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW
        )

        #######################################################

        self.pose_buffer = glGenBuffers(1)

        glBindBuffer(GL_ARRAY_BUFFER, self.pose_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.poses.nbytes, self.poses, GL_DYNAMIC_DRAW)

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 4 * 4 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * 4 * 4, ctypes.c_void_p(16))
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4 * 4 * 4, ctypes.c_void_p(32))
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * 4 * 4, ctypes.c_void_p(48))

        glVertexAttribDivisor(1, 1)
        glVertexAttribDivisor(2, 1)
        glVertexAttribDivisor(3, 1)
        glVertexAttribDivisor(4, 1)

        #######################################################

        glBindVertexArray(0)

    def _setup_shader(self):
        vertex_src = """#version 430 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in mat4 instancePose;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * instancePose * vec4(aPos, 1.0);
}
"""
        fragment_src = """#version 430 core
out vec4 FragColor;

void main(){ 
    FragColor = vec4(c0, c1, c2, c3);
}
"""
        for i in range(4):
            fragment_src = fragment_src.replace(f"c{i}", f"{self.color[i]}")

        self.shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER),
        )

    def _get_uniforms(self):
        self.use()
        self.view_loc = glGetUniformLocation(self.shader, "view")
        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.projection_loc = glGetUniformLocation(self.shader, "projection")

    def set_uniforms(self, model, view, projection):
        self.use()
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(self.projection_loc, 1, GL_FALSE, projection)

    def add_pose(self, pose):

        self.poses_to_add.append(np.ravel(pose))

        glBindBuffer(GL_ARRAY_BUFFER, self.pose_buffer)

        while self.poses_to_add:

            pose_data = self.poses_to_add.pop()
            pose_data = np.asarray(pose_data, dtype=np.float32)
            glBufferSubData(
                GL_ARRAY_BUFFER, int(self.count * 4 * 4 * 4), 4 * 4 * 4, pose_data
            )
            self.count += 1

        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def use(self):
        glUseProgram(self.shader)

    def draw(self, last_n=-1):
        if last_n < 0:
            last_n = self.count
        else:
            last_n = min(last_n, self.count)
        self.use()
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.pose_buffer)
        glDrawElementsInstancedBaseInstance(
            GL_LINE_STRIP,
            self.n_indices,
            GL_UNSIGNED_INT,
            None,
            last_n,
            self.count - last_n,
        )
        glBindVertexArray(0)


class GazeVisualizer(PoseVisualizer):

    MAX_POSES = 100000

    def __init__(
        self,
        field_of_view=45.0,
        aspect_ratio=1.0,
        z_depth=0.1,
        color=[1.0, 1.0, 1.0, 1.0],
    ):

        self.field_of_view = field_of_view
        self.aspect_ratio = aspect_ratio
        self.z_depth = z_depth
        self.color = color

        self.poses = np.tile(np.ravel(np.eye(4, dtype=np.float32)), self.MAX_POSES)
        self.count = 0
        self.poses_to_add = []

        self._setup_vertices()
        self._setup_vertex_arrays()
        self._setup_shader()
        self._get_uniforms()

    def _setup_vertices(self):

        self.vertices = self.z_depth * np.asarray(
            [
                [0, 0, 0],
                [0, 0, -1.0],
            ], 
            dtype=np.float32
        )
        self.vertices = np.ravel(self.vertices)
        
        self.indices = np.asarray([0, 1], dtype=np.uint32)

        self.n_indices = len(self.indices)
