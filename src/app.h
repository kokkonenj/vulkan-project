#pragma once

#include <vk_types.h>
#include <vector>

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
	VkPhysicalDevice gpu;
	VkDevice device;
	VkSurfaceKHR surface;

	VkSwapchainKHR swapchain;
	VkFormat swapchainImageFormat;
	std::vector<VkImage> swapchainImages;
	std::vector<VkImageView> swapchainImageViews;

	VkQueue graphicsQueue;
	uint32_t graphicsQueueFamily;

	VkCommandPool commandPool;
	VkCommandBuffer mainCommandBuffer;

	VkRenderPass renderPass;
	std::vector<VkFramebuffer> frameBuffers;

	VkSemaphore presentSemaphore, renderSemaphore;
	VkFence renderFence;

	void initVulkan();
	void initSwapchain();
	void initCommands();
	void initDefaultRenderpass();
	void initFramebuffers();
	void initSyncStructures();

	void draw();
};