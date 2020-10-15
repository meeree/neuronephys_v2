#version 450 core

layout (location=0) in vec3 position;
layout (location=1) in vec3 color;

layout (location=2) uniform mat4 pMat;
layout (location=3) uniform mat4 vMat;
layout (location=4) uniform mat4 mMat;

out VS_OUT
{   
    vec3 position;
    vec3 color;
} vs_out;

void main(void)
{
    gl_Position = pMat*vMat*mMat*vec4(position, 1.0);
    vs_out.position = position; 
    vs_out.color = color;
}
