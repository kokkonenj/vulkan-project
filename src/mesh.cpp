#include "mesh.h"
#include "utils.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <iostream>
#include <unordered_map>
#include <chrono>

template <>
struct std::hash<Vertex>
{
	size_t operator()(const Vertex& vertex) const
	{
		return (((hash<glm::vec3>()(vertex.position) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.uv) << 1) ^
                (hash<glm::vec3>()(vertex.normal) << 1));
	}
};

VertexInputDescription Vertex::getVertexDescription()
{
	VertexInputDescription description;

	VkVertexInputBindingDescription mainBinding = {};
	mainBinding.binding = 0;
	mainBinding.stride = sizeof(Vertex);
	mainBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	description.bindings.push_back(mainBinding);

	// Position stored at location = 0
	VkVertexInputAttributeDescription positionAttribute = {};
	positionAttribute.binding = 0;
	positionAttribute.location = 0;
	positionAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	positionAttribute.offset = offsetof(Vertex, position);

	// Normal stored at location = 1
	VkVertexInputAttributeDescription normalAttribute = {};
	normalAttribute.binding = 0;
	normalAttribute.location = 1;
	normalAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	normalAttribute.offset = offsetof(Vertex, normal);

	// Color stored at location = 2
	VkVertexInputAttributeDescription colorAttribute = {};
	colorAttribute.binding = 0;
	colorAttribute.location = 2;
	colorAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	colorAttribute.offset = offsetof(Vertex, color);

	// UV coordinates stored at location = 3
	VkVertexInputAttributeDescription uvAttribute = {};
	uvAttribute.binding = 0;
	uvAttribute.location = 3;
	uvAttribute.format = VK_FORMAT_R32G32_SFLOAT;
	uvAttribute.offset = offsetof(Vertex, uv);

	// tangent stored at location = 4
	VkVertexInputAttributeDescription tangentAttribute = {};
	tangentAttribute.binding = 0;
	tangentAttribute.location = 4;
	tangentAttribute.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	tangentAttribute.offset = offsetof(Vertex, tangent);

	description.attributes.push_back(positionAttribute);
	description.attributes.push_back(normalAttribute);
	description.attributes.push_back(colorAttribute);
	description.attributes.push_back(uvAttribute);
	description.attributes.push_back(tangentAttribute);
	return description;
}

bool Mesh::loadFromObj(const char* filename)
{
	auto start = std::chrono::system_clock::now();
	tinyobj::ObjReaderConfig readerConfig;
	readerConfig.mtl_search_path = "../../assets/";
	tinyobj::ObjReader reader;

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	if (!reader.ParseFromFile(filename, readerConfig))
	{
		if (!reader.Error().empty())
		{
			std::cout << "ERROR: " << reader.Error();
			return false;
		}
	}

	if (!reader.Warning().empty())
	{
		std::cout << "WARNING: " << reader.Warning();
	}

	attrib = reader.GetAttrib();
	shapes = reader.GetShapes();
	materials = reader.GetMaterials();

	std::unordered_map<Vertex, uint32_t> uniqueVertices{};
	// loop shapes
	for (size_t s = 0; s < shapes.size(); s++)
	{
		// loop over faces
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
		{
			// load into triangles
			int fv = 3;
			// loop over vertices in face
			for (size_t v = 0; v < fv; v++)
			{
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

				// vertex position
				tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
				tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
				tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

				// vertex normal
				tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
				tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
				tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];

				// vertex uv
				tinyobj::real_t ux = attrib.texcoords[2 * idx.texcoord_index + 0];
				tinyobj::real_t uy = attrib.texcoords[2 * idx.texcoord_index + 1];

				// copy into own vertex struct
				Vertex newVertex = {};
				newVertex.position.x = vx;
				newVertex.position.y = vy;
				newVertex.position.z = vz;
				newVertex.normal.x = nx;
				newVertex.normal.y = ny;
				newVertex.normal.z = nz;
				newVertex.uv.x = ux;
				newVertex.uv.y = 1-uy;

				// set color as offwhite for now
				newVertex.color = glm::vec3(0.95f, 0.95f, 1.0f);

				// save into buffers
				auto res = uniqueVertices.insert({newVertex, static_cast<uint32_t>(vertices.size())});
				if (res.second) {
					vertices.push_back(newVertex);
				}
				indices.push_back(res.first->second);
			}
			index_offset += fv;
		}
	}

	// calculate tangents
	for (size_t t = 0; t < indices.size(); t+=3)
	{
		size_t id1 = indices.at(t + 0);
		size_t id2 = indices.at(t + 1);
		size_t id3 = indices.at(t + 2);
		
		Vertex& v1 = vertices.at(id1);
		Vertex& v2 = vertices.at(id2);
		Vertex& v3 = vertices.at(id3);

		glm::vec3 edge1 = glm::vec3(v2.position - v1.position);
		glm::vec3 edge2 = glm::vec3(v3.position - v1.position);

		float du1 = v2.uv.x - v1.uv.x;
		float dv1 = v2.uv.y - v1.uv.y;

		float du2 = v3.uv.x - v1.uv.x;
		float dv2 = v3.uv.y - v1.uv.y;

		float div = (du1 * dv2 - du2 * dv1);
		float fc = 1.f / div;

		glm::vec3 tang = {
			fc * (dv2 * edge1.x - dv1 * edge2.x),
			fc * (dv2 * edge1.y - dv1 * edge2.y),
			fc * (dv2 * edge1.z - dv1 * edge2.z) };
		tang = glm::normalize(tang);

		float sx = du1, sy = du2;
		float tx = dv1, ty = dv2;
		float handedness = ((tx * sy - ty * sx) < 0.f) ? -1.f : 1.f;
		glm::vec4 t4 = glm::vec4(tang, handedness);
		v1.tangent = t4;
		v2.tangent = t4;
		v3.tangent = t4;
	}
	
	auto end = std::chrono::system_clock::now();
	std::cout << "Loaded model at " << filename << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
	std::cout << "Vertex count: " << vertices.size() << "\n" << "Index count: " << indices.size() << std::endl;
	return true;
}
