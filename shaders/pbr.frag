#version 460

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 texCoord;
layout (location = 2) in vec3 vertPos;
layout (location = 3) in vec3 normal;
layout (location = 4) in vec3 camPos;
layout (location = 5) in vec4 inTangent;
layout (location = 6) in vec3 inLightVec;
layout (location = 7) in vec3 inWorldPos;
layout (location = 8) in vec3 inLightPos;

layout (location = 0) out vec4 outColor;

layout (set = 0, binding = 1) uniform SceneData
{
	vec4 lightPosition;
	vec4 lightColor;
	vec4 ambientColor;
	vec4 sunlightDirection;
	vec4 sunlightColor;
} sceneData;

const float PI = 3.14159265;

layout (set = 0, binding = 2) uniform samplerCube shadowCubeMap;
layout (set = 2, binding = 0) uniform sampler2D albedoMap;
layout (set = 2, binding = 1) uniform sampler2D metallicMap;
layout (set = 2, binding = 2) uniform sampler2D roughnessMap;
layout (set = 2, binding = 3) uniform sampler2D normalMap;
layout (set = 2, binding = 4) uniform sampler2D aoMap;

vec3 getNormalfromMap()
{
	vec3 tangent = inTangent.xyz;
	tangent = (tangent - dot(tangent, normal) * normal);
	vec3 bitangent = cross(normal, inTangent.xyz) * inTangent.w;
	mat3 TBN = mat3(tangent, bitangent, normal);
	vec3 localNormal = 2.0 * texture(normalMap, texCoord).rgb - 1.0;
	return normalize(TBN * localNormal);	
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;
	float num = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;
	return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
	float r = roughness + 1.0;
	float k = (r*r) / 8.0;
	float num = NdotV;
	float denom = NdotV * (1.0 - k) + k;
	return num/denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
	float NdotV = max(dot(N, V), 0.0);
	float NdotL	= max(dot(N, L), 0.0);
	float ggx2 = GeometrySchlickGGX(NdotV, roughness);
	float ggx1 = GeometrySchlickGGX(NdotL, roughness);
	return ggx1 * ggx2;
}

float calculateShadow(vec4 shadowCoord)
{
	return 0.0;
}

void main()
{
	vec3 albedo = pow(texture(albedoMap, texCoord).rgb, vec3(2.2));
	float metallic = texture(metallicMap, texCoord).r;
	float roughness = texture(roughnessMap, texCoord).r;
	float ao = texture(aoMap, texCoord).r;

	vec3 N = getNormalfromMap();
	vec3 V = normalize(camPos - vertPos);

	vec3 F0 = vec3(0.04);
	F0 = mix(F0, albedo, metallic);

	// reflectance
	vec3 Lo = vec3(0.0);

	// go over all lights
	vec3 L = normalize(vec3(sceneData.lightPosition) - vertPos);
	// overwrite for directional light
	//L = normalize(vec3(10.0, 10.0, 5.0));
	vec3 H = normalize(V + L);
	float distance = length(vec3(sceneData.lightPosition) - vertPos);
	float attenuation = 1.0 / (distance * distance);
	// overwrite for directional light
	//attenuation = 0.05;
	vec3 radiance = vec3(sceneData.lightColor) * attenuation;

	float NDF = DistributionGGX(N, H, roughness);
	float G = GeometrySmith(N, V, L, roughness);
	vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

	vec3 kS = F;
	vec3 kD = vec3(1.0) - kS;
	kD *= 1.0 - metallic;

	vec3 num = NDF * G * F;
	float denom = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
	vec3 specular = num / denom;

	float NdotL = max(dot(N, L), 0.0);
	Lo += (kD * albedo / PI + specular) * radiance * NdotL;

	// shadows
	vec3 lightVec = inWorldPos - inLightPos;
	float sampledDist = texture(shadowCubeMap, lightVec).r;
	float dist = length(lightVec);
	float shadow = (dist <= sampledDist + 0.15) ? 1.0 : 0.0;

	vec3 ambient = vec3(sceneData.ambientColor) * albedo * ao;
	vec3 color = ambient + Lo * shadow;

	color = color / (color + vec3(1.0));
	color = pow(color, vec3(1.0/2.2));

	outColor = vec4(color, 1.0);
}