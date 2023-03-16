from pathlib import Path

import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image

from pyflux.colormap import generate_colormap_glsl_code


class DepthTextureShader:
    def __init__(self, llc=0.5, urc=1.0, cm="gist_ncar", cm_levels=1000):

        self.quad_vertices = np.asarray(
            [
                llc,
                urc,
                0.0,
                1.0,
                llc,
                llc,
                0.0,
                0.0,
                urc,
                llc,
                1.0,
                0.0,
                llc,
                urc,
                0.0,
                1.0,
                urc,
                llc,
                1.0,
                0.0,
                urc,
                urc,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )

        self._compile(cm, cm_levels)
        self._get_uniforms()
        self._setup_vertex_array()

    def _setup_vertex_array(self):

        self.VAO = glGenVertexArrays(1)

        self.VBO = glGenBuffers(1)
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(
            GL_ARRAY_BUFFER,
            self.quad_vertices.nbytes,
            self.quad_vertices,
            GL_STATIC_DRAW,
        )

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0,
            2,
            GL_FLOAT,
            GL_FALSE,
            16,
            ctypes.c_void_p(0),
        )
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            1,
            2,
            GL_FLOAT,
            GL_FALSE,
            16,
            ctypes.c_void_p(8),
        )

        glBindVertexArray(0)

    def _compile(self, cm="flag", cm_levels=1000):

        vertex_src = """#version 430 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

void main()
{
    TexCoords = aTexCoords;
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0); 
}
"""

        fragment_src = """#version 430 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D depthMap;

uniform float near_plane;
uniform float far_plane;

float LinearizeDepth(float depth)
{
    float z = depth * 2.0 - 1.0; // Back to NDC 
    return (2.0 * near_plane * far_plane) / (far_plane + near_plane - z * (far_plane - near_plane)) / far_plane;
}

##colormapcode##

void main()
{
    float depthValue = texture(depthMap, TexCoords).r;
    FragColor = get_color(LinearizeDepth(depthValue)/2.0);
    FragColor.w = 1.0;
}
"""
        fragment_src = fragment_src.replace(
            "##colormapcode##", generate_colormap_glsl_code(cm, cm_levels)
        )

        self.shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER),
        )

    def use(self):
        glUseProgram(self.shader)

    def draw(self, depth_texture):
        self.use()
        glBindVertexArray(self.VAO)
        glBindTexture(GL_TEXTURE_2D, depth_texture)
        glDrawArrays(GL_TRIANGLES, 0, 6)

    def _get_uniforms(self):
        self.use()
        self.near_plane_loc = glGetUniformLocation(self.shader, "near_plane")
        self.far_plane_loc = glGetUniformLocation(self.shader, "far_plane")

    def _set_uniforms(self, near_plane, far_plane):
        self.use()
        glUniform1f(self.near_plane_loc, near_plane)
        glUniform1f(self.far_plane_loc, far_plane)


class ShadowMapper:
    def __init__(self, width=800, height=800):
        self._compile()
        self._get_uniforms()

        self.height = height
        self.width = width
        self._setup_texture()

    def _setup_texture(self):
        self.frame_buffer = glGenFramebuffers(1)
        self.depth_map = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.depth_map)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_DEPTH_COMPONENT32,
            self.width,
            self.height,
            0,
            GL_DEPTH_COMPONENT,
            GL_FLOAT,
            None,
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        glBindFramebuffer(GL_FRAMEBUFFER, self.frame_buffer)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depth_map, 0
        )
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _compile(self):

        vertex_src = """#version 430 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 lightSpaceMatrix;


void main()
{
    gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
}
"""
        fragment_src = """#version 430 core
void main(){
    gl_FragDepth = gl_FragCoord.z;
}
"""
        self.shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER),
        )

    def _get_uniforms(self):
        self.use()
        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.lsm_loc = glGetUniformLocation(self.shader, "lightSpaceMatrix")

    def _set_uniforms(self, model, lsm):
        self.use()
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(self.lsm_loc, 1, GL_FALSE, lsm)

    def use(self):
        glUseProgram(self.shader)


class NormalShader:
    def __init__(self):

        self.shader = None
        self._compile()
        self._get_uniforms()

    def _compile(self, cm="jet", cm_levels=1000):

        vertex_src = """#version 430 core
layout (location = 0) in vec3 aPos;
layout (location = 2) in vec3 aNormal;

out VS_OUT {
    vec3 normal;
} vs_out;

uniform mat4 view;
uniform mat4 model;

void main()
{
    gl_Position = view * model * vec4(aPos, 1.0); 
    mat3 normalMatrix = mat3(transpose(inverse(view * model)));
    vs_out.normal = normalize(vec3(vec4(normalMatrix * aNormal, 0.0)));
}
"""

        fragment_src = """#version 430 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0, 1.0, 0.0, 1.0);
}
"""

        geometry_src = """#version 430 core
layout (triangles) in;
layout (line_strip, max_vertices = 6) out;

in VS_OUT {
    vec3 normal;
} gs_in[];

const float MAGNITUDE = 0.02;
  
uniform mat4 projection;

void GenerateLine(int index)
{
    gl_Position = projection * gl_in[index].gl_Position;
    EmitVertex();
    gl_Position = projection * (gl_in[index].gl_Position + 
                                vec4(gs_in[index].normal, 0.0) * MAGNITUDE);
    EmitVertex();
    EndPrimitive();
}

void main()
{
    GenerateLine(0); // first vertex normal
    GenerateLine(1); // second vertex normal
    GenerateLine(2); // third vertex normal
}
"""

        self.shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(geometry_src, GL_GEOMETRY_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER),
        )

    def _get_uniforms(self):
        self.use()
        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.view_loc = glGetUniformLocation(self.shader, "view")
        self.projection_loc = glGetUniformLocation(self.shader, "projection")

    def _set_uniforms(self, model, view, projection):
        self.use()
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(self.projection_loc, 1, GL_FALSE, projection)

    def use(self):
        glUseProgram(self.shader)


class CircleShader:
    def __init__(
        self,
        color=[1.0, 0.0, 0.0, 0.5],
        linewidth=1.0,
        radius=1.0,
        center=[0, 0],
        ratio=1,
        resolution=25,
    ):

        self.color = color
        self.linewidth = linewidth

        self.vertices = np.asarray(center, dtype=np.float32) + radius * np.asarray(
            [
                [ratio * np.cos(phi), np.sin(phi)]
                for phi in list(np.linspace(0, 2 * np.pi, resolution)) + [0]
            ],
            dtype=np.float32,
        )
        self.vertices = np.ravel(self.vertices)

        self._compile()
        self._setup_vertex_array()

    def _setup_vertex_array(self):

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(
            GL_ARRAY_BUFFER,
            self.vertices.nbytes,
            self.vertices,
            GL_STATIC_DRAW,
        )

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0,
            2,
            GL_FLOAT,
            GL_FALSE,
            8,
            ctypes.c_void_p(0),
        )

        glBindVertexArray(0)

    def _compile(self):
        vertex_src = """#version 430 core
layout (location = 0) in vec2 aPos;

void main()
{
    gl_Position = vec4(aPos, -1.0, 1.0);
}
"""
        fragment_src = f"""#version 430 core
out vec4 FragColor;

void main()
{{
    FragColor = vec4({self.color[0]},{self.color[1]},{self.color[2]},{self.color[3]});
}}
"""
        self.shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER),
        )

    def use(self):
        glUseProgram(self.shader)

    def draw(self):
        current_line_width = glGetFloat(GL_LINE_WIDTH)
        glLineWidth(self.linewidth)

        self.use()
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_LINE_STRIP, 0, len(self.vertices) // 2)

        glLineWidth(current_line_width)


class HeatMapShader:
    def __init__(
        self,
        texfile,
        cm="gist_ncar",
    ):
        self.path = Path(__file__).parent
        self._compile(cm=cm)
        self.texfile = texfile
        self._setup_texture(self.texfile)
        self._get_uniforms()

    def _setup_texture(self, texfile):

        self.texture = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.texture)

        # Set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # Set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # load image
        self.image = Image.open(texfile)
        self.image = self.image.transpose(Image.FLIP_TOP_BOTTOM)
        self.img_data = self.image.convert("RGBA").tobytes()
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            self.image.width,
            self.image.height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            self.img_data,
        )

    def _compile(self, cm, cm_levels=1000):

        with open(self.path / "viz" / "vs.glsl", "r") as ifile:
            vertex_src = ifile.read()

        with open(self.path /  "viz" / "fs.glsl", "r") as ifile:
            fragment_src = ifile.read()

        with open(self.path /  "viz" / "gs.glsl", "r") as ifile:
            geometry_src = str(ifile.read())
            geometry_src = geometry_src.replace(
                "##colormapcode##", generate_colormap_glsl_code(cm, cm_levels)
            )

        self.shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(geometry_src, GL_GEOMETRY_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER),
        )

    def _get_uniforms(self):
        self.use()

        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.pov_view_loc = glGetUniformLocation(self.shader, "pov_view")
        self.global_view_loc = glGetUniformLocation(self.shader, "global_view")
        self.projection_loc = glGetUniformLocation(self.shader, "projection")

        self.viewPos_loc = glGetUniformLocation(self.shader, "viewPos")
        self.lightColor_loc = glGetUniformLocation(self.shader, "lightColor")
        self.lightPos_loc = glGetUniformLocation(self.shader, "lightPos")

        self.strength_loc = glGetUniformLocation(self.shader, "strength")

    def _set_uniforms(self, model, pov_view, global_view, projection, strength=0.001):
        self.use()

        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(self.pov_view_loc, 1, GL_FALSE, pov_view)
        glUniformMatrix4fv(self.global_view_loc, 1, GL_FALSE, global_view)
        glUniformMatrix4fv(self.projection_loc, 1, GL_FALSE, projection)

        glUniform3f(self.viewPos_loc, *(np.linalg.inv(global_view)[3, :3]))
        glUniform3f(self.lightPos_loc, *(np.linalg.inv(global_view)[3, :3]))
        glUniform3f(self.lightColor_loc, 1.0, 1.0, 1.0)

        glUniform1f(self.strength_loc, strength)

    def use(self):
        glUseProgram(self.shader)
