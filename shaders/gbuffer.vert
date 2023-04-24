#version 460

layout (location = 0) in vec4 inPos;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inColor;
layout (location = 3) in vec3 inNormal;

layout(set = 0, binding = 0) uniform CameraBuffer
{
	mat4 view;
	mat4 proj;
	mat4 viewproj;
} cameraData;

struct ObjectData
{
	mat4 model;
};

layout(std140, set = 1, binding = 0) readonly buffer ObjectBuffer
{
	ObjectData objects[];
} objectBuffer;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outUV;
layout (location = 2) out vec3 outColor;
layout (location = 3) out vec3 outPos;

void main() 
{
	mat4 modelMatrix = objectBuffer.objects[gl_BaseInstance].model;
	mat4 transformMatrix = (cameraData.viewproj * modelMatrix);
	gl_Position = transformMatrix * inPos;
	
	outUV = inUV;

	// Vertex position in view space
	outPos = vec3(cameraData.view * modelMatrix * inPos);

	// Normal in view space
	mat3 normalMatrix = transpose(inverse(mat3(cameraData.view * modelMatrix)));
	outNormal = normalMatrix * inNormal;

	outColor = inColor;
}