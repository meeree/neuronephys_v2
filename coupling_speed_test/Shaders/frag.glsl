#version 450 core

out vec4 color;

layout (location=2) uniform mat4 pMat;
layout (location=3) uniform mat4 vMat;
layout (location=4) uniform mat4 mMat;
uniform int render_lines;

in VS_OUT
{   
    vec3 position;
    flat float voltage;
} fs_in;

void main(void)
{
    if(render_lines == 1)
    {
        if(fs_in.voltage < 0.5)
            color = vec4(0.9, 0.0, 0.0, 0.9);
        else if(fs_in.voltage < 1.5)
            color = vec4(0.0, 0.0, 0.9, 0.9);
        else if(fs_in.voltage < 2.5)
            color = vec4(0.0, 0.9, 0.9, 0.9);
        else if(fs_in.voltage < 3.5)
            color = vec4(0.0, 0.9, 0.0, 0.9);
        else 
            color = vec4(0.9, 0.9, 0.9, 0.9);
        return;
    }
    color = mix(vec4(1.0, 0.0, 1.0, 1.0), vec4(1.0, 1.0, 0.0, 1.0), (fs_in.voltage + 60) / 15);

    return;
    if((fs_in.voltage + 65) / 2 > 0.8)
        discard;
    color = mix(vec4(1.0, 0.0, 1.0, 1.0), vec4(1.0, 1.0, 0.0, 1.0), (fs_in.voltage + 65) / 2);
}