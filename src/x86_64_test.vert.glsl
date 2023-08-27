#version 450

layout(push_constant) uniform Constants
{
    mat4 view_projection;
} constants;

layout(location = 0) out Out
{
    vec3 uv;
} out_data;

layout(location = 0) in vec3 position; 

void main() 
{
    vec3 vertex_position = position;

    out_data.uv = vertex_position;
    gl_Position = constants.view_projection * vec4(vertex_position, 1.0);
    gl_Position = vec4(gl_Position.x, gl_Position.y, 0, gl_Position.w);
}