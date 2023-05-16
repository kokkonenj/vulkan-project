#version 460

layout (location = 0) out float outFragColor;

layout (location = 0) in vec4 inPos;
layout (location = 1) in vec3 inLightPos;

void main()
{
	// store distance to light
	vec3 lightVec = inPos.xyz - inLightPos;
	outFragColor = length(lightVec);
}