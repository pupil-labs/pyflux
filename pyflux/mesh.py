import numpy as np
import trimesh
from OpenGL.GL import *


def load_ply(path, subdivisions=5):
    mesh = trimesh.load(path)
    meshes = {}
    for _ in range(subdivisions):
        mesh = mesh.subdivide()
    idxs = np.ravel(mesh.faces)
    vertices = mesh.vertices[idxs]
    normals = mesh.vertex_normals[idxs]
    uvs = mesh.visual.uv[idxs]
    color = np.ones_like(normals) * 0.7
    ids = np.arange(len(vertices))
    ids.shape = -1, 1
    temp = np.hstack((vertices, uvs, normals, color, ids))
    meshes["0"] = np.asarray(temp, dtype=np.float32)
    return meshes


def load_glb(path, subdivisions=5):
    scene = trimesh.load(path)
    meshes = {}
    for key in scene.geometry.keys():
        for _ in range(subdivisions):
            scene.geometry[key] = scene.geometry[key].subdivide()
        idxs = np.ravel(scene.geometry[key].faces)
        vertices = scene.geometry[key].vertices[idxs]
        normals = scene.geometry[key].vertex_normals[idxs]
        uvs = scene.geometry[key].visual.uv[idxs]
        color = np.ones_like(normals) * 0.7
        ids = np.arange(len(vertices))
        ids.shape = -1, 1
        temp = np.hstack((vertices, uvs, normals, color, ids))
        meshes[key] = np.asarray(temp, dtype=np.float32)
    return meshes


def load_obj(path):
    _, buffer = ObjLoader.load_model(path)
    vertex_array = np.reshape(buffer, (-1, 8))
    n_triangles = len(vertex_array) // 3
    temp = np.arange(len(vertex_array))
    temp.shape = -1, 1
    vertex_array = np.hstack([vertex_array, temp])
    vertex_array = np.asarray(vertex_array, dtype=np.float32)
    buffer = np.ravel(vertex_array)
    return n_triangles, buffer


class ObjLoader:
    buffer = []

    @staticmethod
    def search_data(data_values, coordinates, skip, data_type):
        for d in data_values:
            if d == skip:
                continue
            if data_type == "float":
                coordinates.append(float(d))
            elif data_type == "int":
                coordinates.append(int(d) - 1)

    @staticmethod  # sorted vertex buffer for use with glDrawArrays function
    def create_sorted_vertex_buffer(indices_data, vertices, textures, normals):
        for i, ind in enumerate(indices_data):
            if i % 3 == 0:  # sort the vertex coordinates
                start = ind * 3
                end = start + 3
                ObjLoader.buffer.extend(vertices[start:end])
            elif i % 3 == 1:  # sort the texture coordinates
                start = ind * 2
                end = start + 2
                ObjLoader.buffer.extend(textures[start:end])
            elif i % 3 == 2:  # sort the normal vectors
                start = ind * 3
                end = start + 3
                ObjLoader.buffer.extend(normals[start:end])

    @staticmethod  # TODO unsorted vertex buffer for use with glDrawElements function
    def create_unsorted_vertex_buffer(indices_data, vertices, textures, normals):
        num_verts = len(vertices) // 3

        for i1 in range(num_verts):
            start = i1 * 3
            end = start + 3
            ObjLoader.buffer.extend(vertices[start:end])

            for i2, data in enumerate(indices_data):
                if i2 % 3 == 0 and data == i1:
                    start = indices_data[i2 + 1] * 2
                    end = start + 2
                    ObjLoader.buffer.extend(textures[start:end])

                    start = indices_data[i2 + 2] * 3
                    end = start + 3
                    ObjLoader.buffer.extend(normals[start:end])

                    break

    @staticmethod
    def show_buffer_data(buffer):
        for i in range(len(buffer) // 8):
            start = i * 8
            end = start + 8
            print(buffer[start:end])

    @staticmethod
    def load_model(file, sorted=True):
        vert_coords = []  # will contain all the vertex coordinates
        tex_coords = []  # will contain all the texture coordinates
        norm_coords = []  # will contain all the vertex normals

        all_indices = []  # will contain all the vertex, texture and normal indices
        indices = []  # will contain the indices for indexed drawing

        with open(file, "r") as f:
            line = f.readline()
            while line:
                values = line.split()
                if values[0] == "v":
                    ObjLoader.search_data(values, vert_coords, "v", "float")
                elif values[0] == "vt":
                    ObjLoader.search_data(values, tex_coords, "vt", "float")
                elif values[0] == "vn":
                    ObjLoader.search_data(values, norm_coords, "vn", "float")
                elif values[0] == "f":
                    for value in values[1:]:
                        val = value.split("/")
                        ObjLoader.search_data(val, all_indices, "f", "int")
                        indices.append(int(val[0]) - 1)

                line = f.readline()

        if sorted:
            # use with glDrawArrays
            ObjLoader.create_sorted_vertex_buffer(
                all_indices, vert_coords, tex_coords, norm_coords
            )
        else:
            # use with glDrawElements
            ObjLoader.create_unsorted_vertex_buffer(
                all_indices, vert_coords, tex_coords, norm_coords
            )

        # ObjLoader.show_buffer_data(ObjLoader.buffer)

        buffer = (
            ObjLoader.buffer.copy()
        )  # create a local copy of the buffer list, otherwise it will overwrite the static field buffer
        ObjLoader.buffer = []  # after copy, make sure to set it back to an empty list

        return np.array(indices, dtype="uint32"), np.array(buffer, dtype="float32")


class HeatTriMesh:
    def __init__(self, vertices, pose=np.eye(4)):

        self.vertices = vertices

        ###### Tansform #TODO: Schoener machen

        extrinsics = np.linalg.inv(pose)

        vertices_ext = np.hstack((vertices[:, :3], np.ones((vertices.shape[0], 1))))
        normals_ext = np.hstack((vertices[:, 5:8], np.zeros((vertices.shape[0], 1))))

        vertices_new = np.einsum("ij,kj->ik", vertices_ext, extrinsics)[:, :3]
        normals_new = np.einsum("ij,kj->ik", normals_ext, extrinsics)[:, :3]

        self.vertices[:, :3] = vertices_new
        self.vertices[:, 5:8] = normals_new

        ###### Tansform

        self.n_triangles = len(vertices) // 3
        self.heatmap = np.zeros(self.n_triangles * 3, dtype=np.float32)

        self._setup_VAO()
        self._setup_SSBO()

    def _setup_VAO(self):

        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)

        buffer = np.ravel(self.vertices)

        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, buffer.nbytes, buffer, GL_STATIC_DRAW)

        offset = 0
        for i, n in enumerate([3, 2, 3, 3, 1]):
            glEnableVertexAttribArray(i)
            glVertexAttribPointer(
                i, n, GL_FLOAT, GL_FALSE, buffer.itemsize * 12, ctypes.c_void_p(offset)
            )
            offset += n * 4

        glBindVertexArray(0)

    def _setup_SSBO(self):

        self.SSBO = glGenBuffers(1)

        heatmap = np.zeros(self.n_triangles * 3, dtype=np.float32)
        self.SSBO_size = heatmap.nbytes
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.SSBO)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.SSBO_size, heatmap, GL_DYNAMIC_COPY)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.SSBO)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def reset_heatmap(self):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.SSBO)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.heatmap.nbytes, self.heatmap)

    def draw_gl(self, shader, primitive=GL_TRIANGLES, ssbo_slot=3):
        glUseProgram(shader)
        glBindVertexArray(self.VAO)
        if ssbo_slot > 0:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.SSBO)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssbo_slot, self.SSBO)
        glDrawArrays(primitive, 0, self.n_triangles * 3)

    def get_heatmap_to_GPU(self):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.SSBO)
        self.heatmap = np.frombuffer(
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.SSBO_size),
            dtype=np.float32,
        )
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
