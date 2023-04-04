#pragma once

#include "vk_types.h"
#include "vk_initializers.h"
#include "vk_pipeline.h"
#include "mesh.h"
#include <vector>
#include <deque>
#include <functional>
#include <unordered_map>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

struct MeshPushConstants
{
	glm::vec3 cameraPosition;
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

struct Texture
{
	AllocatedImage image;
	VkImageView imageView;
};

struct Material
{
	VkDescriptorSet textureSet{ VK_NULL_HANDLE };
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
};

struct PBRMaterial
{
	Texture albedo;
	Texture metallic;
	Texture roughness;
	Texture normal;
	Texture ao;
};

struct RenderObject
{
	Mesh* mesh;
	Material* material;
	glm::mat4 transformMatrix;
};

struct FrameData
{
	VkSemaphore presentSemaphore, renderSemaphore;
	VkFence renderFence;

	VkCommandPool commandPool;
	VkCommandBuffer mainCommandBuffer;

	AllocatedBuffer cameraBuffer;
	VkDescriptorSet globalDescriptor;

	AllocatedBuffer objectBuffer;
	VkDescriptorSet objectDescriptor;

	AllocatedBuffer blurBuffer;
	VkDescriptorSet blurDescriptor;
};

struct GPUBlurData
{
	bool direction;
};

struct GPUCameraData
{
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewproj;
};

struct GPUSceneData
{
	glm::vec4 lightPosition;
	glm::vec4 lightColor;
	glm::vec4 ambientColor;
	glm::vec4 sunlightDirection;
	glm::vec4 sunlightColor;
};

struct GPUObjectData
{
	glm::mat4 modelMatrix;
};

struct UploadContext
{
	VkFence uploadFence;
	VkCommandPool commandPool;
	VkCommandBuffer commandBuffer;
};

constexpr unsigned int FRAME_OVERLAP = 2;

class App {
public:
	App();
	App(const App&) = delete; // delete copy constructor
	App& operator= (const App&) = delete; // delete copy assignment operator
	~App();
	void run();
	AllocatedBuffer createBuffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	void immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function);
	VmaAllocator allocator;
	DeletionQueue mainDeletionQueue;
	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
private:
	bool isInitialized = false;
	int frameNumber = 0;

	struct SDL_Window* window = nullptr;
	VkExtent2D windowExtent = {1280, 720};

	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkPhysicalDevice gpu;
	VkDevice device;
	VkSurfaceKHR surface;

	FrameData frames[FRAME_OVERLAP];

	VkSwapchainKHR swapchain;
	VkFormat swapchainImageFormat;
	std::vector<VkImage> swapchainImages;
	std::vector<VkImageView> swapchainImageViews;

	VkQueue graphicsQueue;
	uint32_t graphicsQueueFamily;

	VkRenderPass renderPass;
	std::vector<VkFramebuffer> frameBuffers;

	// rendering
	std::vector<RenderObject> renderables;
	std::unordered_map<std::string, Material> materials;
	std::unordered_map<std::string, Mesh> meshes;
	GPUSceneData sceneParameters;
	AllocatedBuffer sceneParameterBuffer;
	std::unordered_map<std::string, Texture> loadedTextures;
	std::unordered_map<std::string, PBRMaterial> loadedPBRMaterials;

	// depth resources
	VkImageView depthImageView;
	AllocatedImage depthImage;
	//     format for depth image
	VkFormat depthFormat;

	// MSAA variables
	AllocatedImage colorImage;
	VkImageView colorImageView;

	// post process
	AllocatedImage bloomImage;
	VkImageView bloomImageView;
	VkPipeline blurPipeline;
	VkPipelineLayout bloomPipelineLayout;
	VkDescriptorSetLayout bloomSetLayout;

	VkRenderPass bloomPass;
	VkFramebuffer bloomFramebuffer;

	VkDescriptorPool descriptorPool;
	VkDescriptorSetLayout globalSetLayout;
	VkDescriptorSetLayout objectSetLayout;
	VkDescriptorSetLayout singleTextureSetLayout;
	VkDescriptorSetLayout PBRSetLayout;

	VkPhysicalDeviceProperties gpuProperties;

	UploadContext uploadContext;

	void initVulkan();
	void initSwapchain();
	void initCommands();
	void initDefaultRenderpass();
	void initFramebuffers();
	void initSyncStructures();
	void initPipelines();
	void initScene();

	void draw();

	FrameData& getCurrentFrame();

	bool loadShaderModule(const char* filePath, VkShaderModule* outShaderModule);
	void loadMeshes();
	void loadImages();
	void uploadMesh(Mesh& mesh);

	Material* createMaterial(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name);
	Material* getMaterial(const std::string& name);
	Mesh* getMesh(const std::string& name);
	void drawObjects(VkCommandBuffer commandBuffer, RenderObject* first, int count);

	void initDescriptors();
	size_t padUniformBufferSize(size_t originalSize);

	void initBloomPass();
	void initBloomPipeline();
};