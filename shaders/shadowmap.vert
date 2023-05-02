#version 460

layout (location = 0) in vec3 inPos;

layout (set = 0, binding = 3) uniform UBO
{
	mat4 lightMVP;
} ubo;

void main()
{
	gl_Position = ubo.lightMVP * vec4(inPos, 1.0);
}