#version 450 core

out vec4 color;

uniform sampler2D tex;

in VS_OUT
{   
    vec2 tex_coords;
} fs_in;

void main(void)
{
    float density = texture(tex, fs_in.tex_coords).r;
 //   if(density < 0.01)
 //       discard;
    color = mix(vec4(1.0, 0.0, 0.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0), density); 
}
