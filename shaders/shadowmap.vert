#version 460

layout (location = 0) in vec3 inPos;

layout (location = 0) out vec4 outPos;
layout (location = 1) out vec3 outLightPos;

layout (set = 0, binding = 1) uniform SceneData
{
	vec4 lightPosition;
	vec4 lightColor;
	vec4 ambientColor;
	vec4 sunlightDirection;
	vec4 sunlightColor;
} sceneData;

layout (set = 0, binding = 3) uniform UBO
{
	mat4 model;
	mat4 view;
	mat4 projection;
} ubo;

void main()
{
	gl_Position = ubo.projection * ubo.view * ubo.model * vec4(inPos, 1.0);
	
	outPos = vec4(inPos, 1.0);
	outLightPos = sceneData.lightPosition.xyz;
}