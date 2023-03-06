#pragma once

#include <vk_types.h>
#include <vector>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>

struct VertexInputDescription
{
	std::vector<VkVertexInputBindingDescription> bindings;
	std::vector<VkVertexInputAttributeDescription> attributes;
	VkPipelineVertexInputStateCreateFlags flags = 0;
};

struct Vertex
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
	glm::vec2 uv;
	static VertexInputDescription getVertexDescription();

	bool operator==(const Vertex& other) const
	{
		return position == other.position && normal == other.normal && color == other.color && uv == other.uv;
	}
};

struct Mesh
{
	std::vector<Vertex> vertices;
	AllocatedBuffer vertexBuffer;
	std::vector<uint32_t> indices;
	AllocatedBuffer indexBuffer;

	bool loadFromObj(const char* filename);
};