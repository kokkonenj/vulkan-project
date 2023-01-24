#pragma once

#include <vk_types.h>

class App {
public:
	bool isInitialized{ false };
	int frameNumber{ 0 };

	VkExtent2D windowExtent{ 800, 600 };

	struct SDL_Window* window{ nullptr };

	void init();
	void cleanup();
	void draw();
	void run();
};