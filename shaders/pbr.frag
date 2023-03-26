#version 460

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 texCoord;
layout (location = 2) in vec3 vertPos;
layout (location = 3) in vec3 normal;
layout (location = 4) in vec3 camPos;

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

layout (set = 2, binding = 0) uniform sampler2D albedoMap;
layout (set = 2, binding = 1) uniform sampler2D metallicMap;
layout (set = 2, binding = 2) uniform sampler2D roughnessMap;
layout (set = 2, binding = 3) uniform sampler2D normalMap;
layout (set = 2, binding = 4) uniform sampler2D aoMap;

vec3 getNormalfromMap()
{
	vec3 tangentNormal = normalize(texture(normalMap, texCoord).xyz * 2.0 - 1.0);

	vec3 Q1 = dFdx(vertPos);
	vec3 Q2 = dFdy(vertPos);
	vec2 st1 = dFdx(texCoord);
	vec2 st2 = dFdy(texCoord);

	vec3 N = normalize(normal);
	vec3 T = normalize(Q1*st2.t - Q2*st1.t);
	vec3 B = -normalize(cross(N,T));
	mat3 TBN = mat3(T, B, N);

	return normalize(TBN * tangentNormal);
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

void main()
{
	vec3 albedo = pow(texture(albedoMap, texCoord).rgb, vec3(2.2));
	float metallic = texture(metallicMap, texCoord).r;
	float roughness = texture(roughnessMap, texCoord).r;
	float ao = texture(aoMap, texCoord).r;

	vec3 N = normalize(normal);
	vec3 V = normalize(camPos - vertPos);

	vec3 F0 = vec3(0.04);
	F0 = mix(F0, albedo, metallic);

	// reflectance
	vec3 Lo = vec3(0.0);

	// go over all lights
	vec3 L = normalize(vec3(sceneData.lightPosition) - vertPos);
	vec3 H = normalize(V + L);
	float distance = length(vec3(sceneData.lightPosition) - vertPos);
	float attenuation = 1.0 / (distance * distance);
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

	vec3 ambient = vec3(sceneData.ambientColor) * albedo * ao;
	vec3 color = ambient + Lo;

	color = color / (color + vec3(1.0));
	color = pow(color, vec3(1.0/2.2));

	outColor = vec4(color, 1.0);
}