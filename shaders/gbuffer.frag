#version 460

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inColor;
layout (location = 3) in vec3 inPos;

layout (location = 0) out vec4 outPosition;
layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec4 outAlbedo;

layout (set = 0, binding = 1) uniform SceneData
{
	vec4 lightPosition;
	vec4 lightColor;
	vec4 ambientColor;
	vec4 sunlightDirection;
	vec4 sunlightColor;
} sceneData;

layout (set = 2, binding = 0) uniform sampler2D albedoMap;

float linearDepth(float depth)
{
	float z = depth * 2.0 - 1.0; 
	return (2.0 * 0.1 * 64.0) / (64.0 + 0.1 - z * (64.0 - 0.1));	
}

void main() 
{
	outPosition = vec4(inPos, linearDepth(gl_FragCoord.z));
	outNormal = vec4(normalize(inNormal) * 0.5 + 0.5, 1.0);
	outAlbedo = texture(albedoMap, inUV) * vec4(inColor, 1.0);
}