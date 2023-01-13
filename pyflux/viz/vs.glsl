#version 430 core
layout (location = 0) in vec3 Pos;
layout (location = 1) in vec2 Tex;
layout (location = 2) in vec3 Normal;
layout (location = 3) in vec3 Color;
layout (location = 4) in float id;

out VS_OUT {
    vec2 Tex;
    vec3 FragPos;
    vec3 FragNormal;
    vec3 ViewPos;
    vec3 ViewNormal;
    vec3 Color;
    vec4 pov_gl_Position;
    float id;
} vs_out;

uniform mat4 model;
uniform mat4 pov_view;
uniform mat4 global_view;
uniform mat4 projection;

void main()
{
    vs_out.FragPos = vec3(model * vec4(Pos, 1.0));
    vs_out.FragNormal = mat3(transpose(inverse(model))) * Normal; 
    vs_out.ViewPos = vec3(pov_view * vec4(vs_out.FragPos, 1.0));
    vs_out.ViewNormal = vec3(pov_view * vec4(vs_out.FragNormal, 0.0));
    vs_out.pov_gl_Position = projection * vec4(vs_out.ViewPos, 1.0);
    vs_out.Color = Color;
    vs_out.id = id;
    vs_out.Tex = Tex;

    gl_Position = projection * global_view * model * vec4(Pos, 1.0);
    
}