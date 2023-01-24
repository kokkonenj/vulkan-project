#pragma once

#include <vk_types.h>

class App {
public:
	App();
	App(const App&) = delete; // delete copy constructor
	App& operator= (const App&) = delete; // delete copy assignment operator
	~App();
	void run();
private:
	bool isInitialized = false;
	int frameNumber = 0;

	struct SDL_Window* window = nullptr;
	VkExtent2D windowExtent = {800, 600};

	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkPhysicalDevice physicalDevice;
	VkDevice device;
	VkSurfaceKHR surface;

	void init_vulkan();
};