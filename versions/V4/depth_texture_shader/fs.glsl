#version 430 core
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