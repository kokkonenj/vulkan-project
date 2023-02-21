#pragma once

#include "vk_types.h"
#include "vk_initializers.h"
#include "vk_pipeline.h"
#include "mesh.h"
#include <vector>
#include <deque>
#include <functional>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

struct MeshPushConstants
{
	glm::vec4 data;
	glm::mat4 renderMatrix;
};

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function)
	{
		deletors.push_back(function);
	}

	void flush()
	{
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++)
		{
			(*it)();
		}
		deletors.clear();
	}
};

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

	VmaAllocator allocator;

	VkPipelineLayout trianglePipelineLayout;
	VkPipeline trianglePipeline;
	Mesh triangleMesh;

	VkPipelineLayout meshPipelineLayout;
	VkPipeline meshPipeline;

	DeletionQueue mainDeletionQueue;

	Mesh monkeyMesh;


	void initVulkan();
	void initSwapchain();
	void initCommands();
	void initDefaultRenderpass();
	void initFramebuffers();
	void initSyncStructures();
	void initPipelines();

	void draw();

	bool loadShaderModule(const char* filePath, VkShaderModule* outShaderModule);
	void loadMeshes();
	void uploadMesh(Mesh& mesh);
};