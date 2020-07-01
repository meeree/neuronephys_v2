#version 450 core

out vec4 color;

layout (location=2) uniform mat4 pMat;
layout (location=3) uniform mat4 vMat;
layout (location=4) uniform mat4 mMat;

in VS_OUT
{   
    vec3 position;
    float voltage;
} fs_in;

void main(void)
{
    if(fs_in.voltage < 0.001)
        discard;
    color = mix(vec4(0.4, 0.0, 0.0, 0.8), vec4(0.0, 1.0, 0.0, 1.0), fs_in.voltage); 
}
