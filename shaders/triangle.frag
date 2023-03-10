#version 460

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 texCoord;
layout (location = 2) in vec3 vertPos;
layout (location = 3) in vec3 normal;

layout (location = 0) out vec4 outColor;

layout (set = 0, binding = 1) uniform SceneData
{
	vec4 lightPosition;
	vec4 lightColor;
	vec4 ambientColor;
	vec4 sunlightDirection;
	vec4 sunlightColor;
} sceneData;

void main()
{
	const float lightPower = 5.0;
	const vec3 diffuseColor = vec3(0.3, 0.3, 0.3);
	const vec3 specularColor = vec3(1.0);
	const float shininess = 8.0;
	const float gamma = 2.2;

	vec3 normal = normalize(normal);
	vec3 lightDir = vec3(sceneData.lightPosition.xyz) - vertPos;
	float distance = length(lightDir);
	distance = distance * distance;
	lightDir = normalize(lightDir);
	
	float lambertian = max(dot(lightDir, normal), 0.0);
	float specular = 0.0;
	
	if (lambertian > 0.0)
	{
		vec3 viewDir = normalize(-vertPos);
		// blinn phong
		vec3 halfDir = normalize(lightDir + viewDir);
		float specAngle = max(dot(halfDir, normal), 0.0);
		specular = pow(specAngle, shininess);
	}
	
	vec3 colorLinear = inColor + sceneData.ambientColor.xyz + diffuseColor * lambertian * sceneData.lightColor.xyz * lightPower / distance + specularColor * specular * sceneData.lightColor.xyz * lightPower / distance;
	vec3 gammaCorrected = pow(colorLinear, vec3(1.0/gamma));
	
	outColor = vec4(gammaCorrected, 1.0);
}