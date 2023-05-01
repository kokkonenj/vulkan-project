#pragma once
#include <vk_types.h>
#include <app.h>

namespace utils {
	bool loadImageFromFile(App* app, const char* file, AllocatedImage& outImage, VkFormat format);
}