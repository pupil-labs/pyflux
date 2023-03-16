#version 430 core
#define M_PI 3.1415926535897932384626433832795
#define THRESHOLD 0.984807753012208 //cos(10.0*180.0/M_PI)

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

layout(std430, binding = 3) coherent buffer SSBO {
    float heat[];
};

in VS_OUT {
    vec2 Tex;
    vec3 FragPos;
    vec3 FragNormal;
    vec3 ViewPos;
    vec3 ViewNormal;
    vec3 Color;
    vec4 pov_gl_Position;
    float id;
} gs_in[];

out vec3 FragNormal;  
out vec3 FragPos; 
out vec4 Heat_Color;
out vec3 Color;
out vec2 Tex;

vec3 flux_density(vec3 pos){
    
    //float sigma = 1./30.;
    float sigma = 0.0875/2;
    float h = pos.z;
    float density = 0;
    if (h<0){
        density = 1./(2.* M_PI * pow(sigma*h,2)) * exp(-pow(length(pos.xy),2)/(2.*pow(h*sigma,2)));
    }
    return normalize(pos) * density;

}

##colormapcode##

vec3 flux[3];
vec3 leistung;
vec4 pov_gl_Position;
float integrated_flux;
int id;

layout(binding=0) uniform sampler2D depthMap;  
layout(binding=1) uniform sampler2D texMap;   

uniform float strength;

//uniform sampler2D depthMap;

void main() {  

    for (int i=0;i<3;i++){    

        flux[i] = flux_density(gs_in[i].ViewPos);
        leistung[i] = length(flux[i])*dot(normalize(flux[i]),-gs_in[i].ViewNormal);
    }    
        
    integrated_flux = max(min(strength *1./3.*(leistung[0]+leistung[1]+leistung[2]),1),0); 

    for (int i=0;i<3;i++){    
    
        id = int(gs_in[i].id);
        pov_gl_Position = gs_in[i].pov_gl_Position;
        FragNormal = gs_in[i].FragNormal; 
        FragPos = gs_in[i].FragPos;
        Color = gs_in[i].Color;
        Tex = gs_in[i].Tex;

        if (-gs_in[i].ViewPos.z/length(gs_in[i].ViewPos)>THRESHOLD){

            vec3 clip_pos = (pov_gl_Position.xyz/pov_gl_Position.w+1.0)/2.0;
            float foreground = texture(depthMap, clip_pos.xy).r;

            if (foreground-clip_pos.z>-0.00001){
                heat[id] += 0.1*integrated_flux;                
            }
        }

        Heat_Color  = get_color(heat[id]);
        gl_Position = gl_in[i].gl_Position;

        EmitVertex();

    }
    
    EndPrimitive();

} 