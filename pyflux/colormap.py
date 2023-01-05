import numpy as np
from matplotlib import cm


def generate_colormap_glsl_code(cmap, levels):

    cmap = cm.get_cmap(cmap, levels)
    values = cmap(np.linspace(0, 1, levels))[:, :3]

    glsl_code = f"vec4 get_color(float x){{\nvec4 colors[{levels}];\n"
    for i, x in enumerate(values):
        glsl_code += f"colors[{i}]=vec4({x[0]},{x[1]},{x[2]},{np.sqrt(i/levels)});\n"
    glsl_code += f"int index = min(max(int(x*{levels}),0),{levels-1});\n"
    glsl_code += (
        "if(x>-1.0){return colors[index];}else{return vec4(0.7,0.7,0.7,1.0);}\n}"
    )

    return glsl_code
