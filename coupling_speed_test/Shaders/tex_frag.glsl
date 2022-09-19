#version 450 core

out vec4 color;

uniform sampler2D tex;

in VS_OUT
{   
    vec2 tex_coords;
} fs_in;

vec4 rainbow(float t)
{
    vec4 color = mix(vec4(1.0, 0.0, 0.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0), t / 0.2);
    if(t > 0.2)
        color = mix(vec4(0.0, 1.0, 0.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0), (t - 0.2) / 0.2);
    if(t > 0.4)
        color = mix(vec4(0.0, 0.0, 1.0, 1.0), vec4(1.0, 0.0, 1.0, 1.0), (t - 0.4) / 0.2);
    if(t > 0.6)
        color = mix(vec4(1.0, 0.0, 1.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0), (t - 0.6) / 0.2);
    if(t > 0.8)
        color = mix(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.0, 1.0, 1.0, 1.0), (t - 0.8) / 0.2);
    return color;
}

vec4 purple_to_yellow(float t)
{
    return mix(vec4(1.0, 0.0, 1.0, 1.0), vec4(1.0, 1.0, 0.0, 1.0), t);
}

void main(void)
{
    float density = texture(tex, fs_in.tex_coords).r;
    if(density < 0.00001)
        discard;

    color = purple_to_yellow( 30 * density);
}
