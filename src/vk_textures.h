#pragma once
#include <vk_types.h>
#include <app.h>

namespace utils {
	bool loadImageFromFile(App* app, const char* file, AllocatedImage& outImage, VkFormat format);
	bool loadImageFromBuffer(App* app, void* buffer, AllocatedImage& outImage, VkFormat format, uint32_t texWidth, uint32_t texHeight);
}