#version 430 core
out vec4 FragColor;

in vec3 FragNormal;  
in vec3 FragPos; 
in vec4 Heat_Color; 
in vec3 Color;
in vec2 Tex;
  
uniform vec3 viewPos; 
uniform vec3 lightPos; 
uniform vec3 lightColor;

layout(binding=0) uniform sampler2D depthMap;  
layout(binding=1) uniform sampler2D texMap;   

void main()
{
    // ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
  	
    // diffuse 
    vec3 norm = normalize(FragNormal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // specular
    float specularStrength = 0.2;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec4 texture_color = texture(texMap, Tex);

    vec3 result = (ambient + diffuse + specular) * Heat_Color.xyz;

    // vec3 color = (ambient + diffuse + specular) * Color.xyz;

    // vec3 resulta = (ambient + diffuse + specular) * texture_color.xyz;

    vec3 color = Color.xyz;

    vec3 resulta = texture_color.xyz;

    vec3 resultb = mix(vec4(resulta, 1.0), vec4(color,1.0), 0.5).xyz;
    
    FragColor = mix(vec4(resulta, 1.0), vec4(result,1.0), Heat_Color.w);
    
} 