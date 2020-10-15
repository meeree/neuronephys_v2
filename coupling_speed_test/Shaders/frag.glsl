#version 450 core

out vec4 color;

layout (location=2) uniform mat4 pMat;
layout (location=3) uniform mat4 vMat;
layout (location=4) uniform mat4 mMat;

in VS_OUT
{   
    vec3 position;
    vec3 color;
} fs_in;

void main(void)
{
    color = vec4(fs_in.color[0], fs_in.color[1], fs_in.color[2], 1.0f);
    //color = vec4(1.0f,0.0f,0.0f, 1.0f);
}
