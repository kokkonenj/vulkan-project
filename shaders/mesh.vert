#version 460

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;

layout (location = 0) out vec3 outColor;

layout (push_constant) uniform constants
{
	vec4 data;
	mat4 renderMatrix;
} PushConstants;

void main()
{
	gl_Position = PushConstants.renderMatrix * vec4(aPos, 1.0f);
	outColor = aColor;
}