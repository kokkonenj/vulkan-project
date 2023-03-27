#version 460

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;
layout (location = 3) in vec2 aTexCoord;

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 texCoord;
layout (location = 2) out vec3 vertPos;
layout (location = 3) out vec3 normal;
layout (location = 4) out vec3 camPos;

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

layout (push_constant) uniform constants
{
	vec3 cameraPosition;
} PushConstants;

void main()
{
	mat4 modelMatrix = objectBuffer.objects[gl_BaseInstance].model;
	mat4 transformMatrix = (cameraData.viewproj * modelMatrix);
	gl_Position = transformMatrix * vec4(aPos, 1.0);
	vec4 vertPos4 = modelMatrix * vec4(aPos, 1.0);
	mat4 invView = inverse(cameraData.view);
	camPos = vec3(invView[3][0], invView[3][1], invView[3][2]);
	outColor = aColor;
	texCoord = aTexCoord;
	vertPos = vec3(vertPos4);
	normal = transpose(inverse(mat3(modelMatrix))) * aNormal;
}