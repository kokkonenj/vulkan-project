#include "app.h"
#include <SDL.h>
#include <SDL_vulkan.h>
#include <VkBootstrap.h>
#include <iostream>
#include <fstream>
#include <array>
#include "vk_textures.h"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

// Check for unhandled vulkan errors, and abort if encountered
#define VK_CHECK(x)												\
do																\
{																\
	VkResult err = x;											\
	if (err)													\
	{															\
		std::cerr << "Vulkan error: " << err << std::endl;		\
		std::abort();											\
	}															\
} while (0);													\

// public
App::App()
{
	// Initialize SDL
	SDL_Init(SDL_INIT_VIDEO);
	SDL_WindowFlags windowFlags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

	// Create SDL window
	window = SDL_CreateWindow(
		"vulkan-project",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		windowExtent.width,
		windowExtent.height,
		windowFlags
	);

	initVulkan();
	initSwapchain();
	initCommands();
	initDefaultRenderpass();
	initFramebuffers();
	initCubeMap();
	initShadowPass();
	initShadowPassFramebuffer();
	initSyncStructures();
	initDescriptors();
	initPipelines();
	loadImages();
	loadMeshes();
	initScene();

	isInitialized = true;
}

App::~App()
{
	if (isInitialized)
	{
		vkDeviceWaitIdle(device); // wait for GPU to finish
		mainDeletionQueue.flush();
		vmaDestroyAllocator(allocator);
		vkDestroyDevice(device, nullptr);
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkb::destroy_debug_utils_messenger(instance, debugMessenger);
		vkDestroyInstance(instance, nullptr);
		SDL_DestroyWindow(window);
	}
}

void App::run()
{
	SDL_Event e;
	bool quitSignal = false;
	
	//main loop
	while (!quitSignal)
	{
		while (SDL_PollEvent(&e) != 0)
		{
			if (e.type == SDL_QUIT) quitSignal = true;
		}
		draw();
	}
}

// private
void App::initVulkan()
{
	// Get instance and debug messenger using vkBootstrap
	vkb::InstanceBuilder builder;
	auto instRet = builder.set_app_name("Vulkan App")
		.request_validation_layers(true)
		.require_api_version(1, 1, 0)
		.use_default_debug_messenger()
		.build();
	vkb::Instance vkbInst = instRet.value();
	instance = vkbInst.instance;
	debugMessenger = vkbInst.debug_messenger;

	// Get surface of the window
	SDL_Vulkan_CreateSurface(window, instance, &surface);

	// Select device with vkBootstrap
	vkb::PhysicalDeviceSelector selector{ vkbInst };
	vkb::PhysicalDevice physicalDevice = selector
		.set_minimum_version(1, 1)
		.set_surface(surface)
		.select()
		.value();
	vkb::DeviceBuilder deviceBuilder{ physicalDevice };
	VkPhysicalDeviceShaderDrawParametersFeatures shaderDrawParametersFeatures = {};
	shaderDrawParametersFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES;
	shaderDrawParametersFeatures.pNext = nullptr;
	shaderDrawParametersFeatures.shaderDrawParameters = VK_TRUE;
	vkb::Device vkbDevice = deviceBuilder.add_pNext(&shaderDrawParametersFeatures).build().value();
	device = vkbDevice.device;
	gpu = physicalDevice.physical_device;

	// Get graphics queue with vkBootstrap
	graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

	// Initialize memory allocator
	VmaAllocatorCreateInfo allocatorInfo = {};
	allocatorInfo.physicalDevice = gpu;
	allocatorInfo.device = device;
	allocatorInfo.instance = instance;
	vmaCreateAllocator(&allocatorInfo, &allocator);

	gpuProperties = vkbDevice.physical_device.properties;
	std::cout << "GPU has minimum buffer alignment of: " << gpuProperties.limits.minUniformBufferOffsetAlignment << std::endl;
	VkSampleCountFlags counts = gpuProperties.limits.framebufferColorSampleCounts & gpuProperties.limits.framebufferDepthSampleCounts;
	if (counts & VK_SAMPLE_COUNT_64_BIT) { msaaSamples = VK_SAMPLE_COUNT_64_BIT; }
	else if (counts & VK_SAMPLE_COUNT_32_BIT) { msaaSamples = VK_SAMPLE_COUNT_32_BIT; }
	else if (counts & VK_SAMPLE_COUNT_16_BIT) { msaaSamples = VK_SAMPLE_COUNT_16_BIT; }
	else if (counts & VK_SAMPLE_COUNT_8_BIT) { msaaSamples = VK_SAMPLE_COUNT_8_BIT; }
	else if (counts & VK_SAMPLE_COUNT_4_BIT) { msaaSamples = VK_SAMPLE_COUNT_4_BIT; }
	else if (counts & VK_SAMPLE_COUNT_2_BIT) { msaaSamples = VK_SAMPLE_COUNT_2_BIT; }
	else { msaaSamples = VK_SAMPLE_COUNT_1_BIT; }
	std::cout << "Rendering with " << msaaSamples << "x MSAA" << std::endl;
}

void App::initSwapchain()
{
	// Create swapchain using vkBootstrap
	vkb::SwapchainBuilder swapchainbuilder{ gpu, device, surface };
	vkb::Swapchain vkbSwapchain = swapchainbuilder
		.use_default_format_selection()
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.set_desired_extent(windowExtent.width, windowExtent.height)
		.build()
		.value();
	swapchain = vkbSwapchain.swapchain;
	swapchainImages = vkbSwapchain.get_images().value();
	swapchainImageViews = vkbSwapchain.get_image_views().value();
	swapchainImageFormat = vkbSwapchain.image_format;

	mainDeletionQueue.push_function([=]()
		{
			vkDestroySwapchainKHR(device, swapchain, nullptr);
		});

	// depth image stuff
	VkExtent3D depthImageExtent = {
		windowExtent.width,
		windowExtent.height,
		1
	};
	depthFormat = VK_FORMAT_D32_SFLOAT;

	VkImageCreateInfo depthImageInfo = VkInit::imageCreateInfo(depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImageExtent, msaaSamples);

	// allocate depth image from GPU local memory
	VmaAllocationCreateInfo depthImageAllocInfo = {};
	depthImageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	depthImageAllocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	vmaCreateImage(allocator, &depthImageInfo, &depthImageAllocInfo, &depthImage.image, &depthImage.allocation, nullptr);

	// build image-view for depth image
	VkImageViewCreateInfo depthViewInfo = VkInit::imageviewCreateInfo(depthFormat, depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);
	VK_CHECK(vkCreateImageView(device, &depthViewInfo, nullptr, &depthImageView));

	// colors for MSAA
	VkExtent3D colorImageExtent = { windowExtent.width, windowExtent.height, 1 };
	VkImageCreateInfo colorImageInfo = VkInit::imageCreateInfo(swapchainImageFormat, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, colorImageExtent, msaaSamples);
	VmaAllocationCreateInfo colorImageAllocInfo = {};
	colorImageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	colorImageAllocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	vmaCreateImage(allocator, &colorImageInfo, &colorImageAllocInfo, &colorImage.image, &colorImage.allocation, nullptr);
	VkImageViewCreateInfo colorViewInfo = VkInit::imageviewCreateInfo(swapchainImageFormat, colorImage.image, VK_IMAGE_ASPECT_COLOR_BIT);
	VK_CHECK(vkCreateImageView(device, &colorViewInfo, nullptr, &colorImageView));

	// cleanup
	mainDeletionQueue.push_function([=]()
		{
			vkDestroyImageView(device, colorImageView, nullptr);
			vmaDestroyImage(allocator, colorImage.image, colorImage.allocation);
			vkDestroyImageView(device, depthImageView, nullptr);
			vmaDestroyImage(allocator, depthImage.image, depthImage.allocation);
		});
}

void App::initCommands()
{
	// Create command pool for commands submitted to the graphics queue
	VkCommandPoolCreateInfo commandPoolInfo = VkInit::commandPoolCreateInfo(graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	for (int i = 0; i < FRAME_OVERLAP; i++)
	{
		VK_CHECK(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &frames[i].commandPool));
		
		// allocate default command buffer
		VkCommandBufferAllocateInfo cmdAllocInfo = VkInit::commandBufferallocateInfo(frames[i].commandPool, 1);
		VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &frames[i].mainCommandBuffer));

		mainDeletionQueue.push_function([=]()
			{
				vkDestroyCommandPool(device, frames[i].commandPool, nullptr);
			});
	}
	VkCommandPoolCreateInfo uploadCommandPoolInfo = VkInit::commandPoolCreateInfo(graphicsQueueFamily);
	VK_CHECK(vkCreateCommandPool(device, &uploadCommandPoolInfo, nullptr, &uploadContext.commandPool));
	mainDeletionQueue.push_function([=]()
		{
			vkDestroyCommandPool(device, uploadContext.commandPool, nullptr);
		});

	VkCommandBufferAllocateInfo cmdAllocInfo = VkInit::commandBufferallocateInfo(uploadContext.commandPool, 1);
	VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &uploadContext.commandBuffer));
}

void App::initDefaultRenderpass()
{
	// Defining color attachment for the renderpass
	VkAttachmentDescription colorAttachment = {};
	colorAttachment.format = swapchainImageFormat;
	colorAttachment.samples = msaaSamples;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference colorAttatchmentReference = {};
	colorAttatchmentReference.attachment = 0;
	colorAttatchmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	// depth attachment
	VkAttachmentDescription depthAttachment = {};
	depthAttachment.flags = 0;
	depthAttachment.format = depthFormat;
	depthAttachment.samples = msaaSamples;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthAttachmentReference = {};
	depthAttachmentReference.attachment = 1;
	depthAttachmentReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	// color attachment resolve for MSAA
	VkAttachmentDescription colorAttachmentResolve = {};
	colorAttachmentResolve.format = swapchainImageFormat;
	colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference colorATtachmentResolveReference = {};
	colorATtachmentResolveReference.attachment = 2;
	colorATtachmentResolveReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	// subpass
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttatchmentReference;
	subpass.pDepthStencilAttachment = &depthAttachmentReference;
	subpass.pResolveAttachments = &colorATtachmentResolveReference;

	VkSubpassDependency dependency = {};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	VkSubpassDependency depthDependency = {};
	depthDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	depthDependency.dstSubpass = 0;
	depthDependency.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	depthDependency.srcAccessMask = 0;
	depthDependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	depthDependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

	// dependencies array
	VkSubpassDependency dependencies[2] = { dependency, depthDependency };

	// attachments array
	VkAttachmentDescription attachments[3] = { colorAttachment, depthAttachment, colorAttachmentResolve };

	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = 3;
	renderPassInfo.pAttachments = &attachments[0];
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;
	renderPassInfo.dependencyCount = 2;
	renderPassInfo.pDependencies = &dependencies[0];

	VK_CHECK(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));

	mainDeletionQueue.push_function([=]()
		{
			vkDestroyRenderPass(device, renderPass, nullptr);
		});
}

void App::initFramebuffers()
{
	VkFramebufferCreateInfo frameBufferInfo = {};
	frameBufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	frameBufferInfo.pNext = nullptr;
	frameBufferInfo.renderPass = renderPass;
	frameBufferInfo.attachmentCount = 1;
	frameBufferInfo.width = windowExtent.width;
	frameBufferInfo.height = windowExtent.height;
	frameBufferInfo.layers = 1;

	const size_t swapchainImageCount = swapchainImages.size();
	frameBuffers = std::vector<VkFramebuffer>(swapchainImageCount);

	for (unsigned int i = 0; i < swapchainImageCount; i++)
	{
		VkImageView attachments[3] = { colorImageView, depthImageView, swapchainImageViews[i] };

		frameBufferInfo.pAttachments = attachments;
		frameBufferInfo.attachmentCount = 3;
		VK_CHECK(vkCreateFramebuffer(device, &frameBufferInfo, nullptr, &frameBuffers[i]));

		mainDeletionQueue.push_function([=]()
			{
				vkDestroyFramebuffer(device, frameBuffers[i], nullptr);
				vkDestroyImageView(device, swapchainImageViews[i], nullptr);
			});
	}
}

void App::initSyncStructures()
{
	// Create synchronization structures
	VkFenceCreateInfo fenceCreateInfo = VkInit::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
	VkSemaphoreCreateInfo semaphoreCreateInfo = VkInit::semaphoreCreateInfo();

	for (int i = 0; i < FRAME_OVERLAP; i++)
	{
		VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &frames[i].renderFence));
		mainDeletionQueue.push_function([=]()
			{
				vkDestroyFence(device, frames[i].renderFence, nullptr);
			});
		VK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frames[i].presentSemaphore));
		VK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frames[i].renderSemaphore));
		mainDeletionQueue.push_function([=]()
			{
				vkDestroySemaphore(device, frames[i].presentSemaphore, nullptr);
				vkDestroySemaphore(device, frames[i].renderSemaphore, nullptr);
			});
	}

	VkFenceCreateInfo uploadFenceCreateInfo = VkInit::fenceCreateInfo();
	VK_CHECK(vkCreateFence(device, &uploadFenceCreateInfo, nullptr, &uploadContext.uploadFence));
	mainDeletionQueue.push_function([=]()
		{
			vkDestroyFence(device, uploadContext.uploadFence, nullptr);
		});
}

void App::initPipelines()
{
	VkShaderModule pbrFragShader;
	if (!loadShaderModule("../../shaders/pbr.frag.spv", &pbrFragShader))
	{
		std::cout << "Error loading pbr frag shader" << std::endl;
	}

	VkShaderModule pbrVertShader;
	if (!loadShaderModule("../../shaders/pbr.vert.spv", &pbrVertShader))
	{
		std::cout << "Error when loading the mesh vertex shader module" << std::endl;
	}

	VkShaderModule shadowMapVertShader;
	if (!loadShaderModule("../../shaders/shadowmap.vert.spv", &shadowMapVertShader))
	{
		std::cout << "Error when loading the shadowmap vertex shader module" << std::endl;
	}

	VkShaderModule shadowMapFragShader;
	if (!loadShaderModule("../../shaders/shadowmap.frag.spv", &shadowMapFragShader))
	{
		std::cout << "Error when loading the shadowmap fragment shader module" << std::endl;
	}

	// push constants
	VkPushConstantRange pushConstants;
	pushConstants.offset = 0;
	pushConstants.size = sizeof(MeshPushConstants);
	pushConstants.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	PipelineBuilder pipelineBuilder;
	pipelineBuilder.vertexInputInfo = VkInit::vertexInputStateCreateInfo();
	pipelineBuilder.inputAssembly = VkInit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	pipelineBuilder.viewport.x = 0.0f;
	pipelineBuilder.viewport.y = 0.0f;
	pipelineBuilder.viewport.width = (float)windowExtent.width;
	pipelineBuilder.viewport.height = (float)windowExtent.height;
	pipelineBuilder.viewport.minDepth = 0.0f;
	pipelineBuilder.viewport.maxDepth = 1.0f;
	pipelineBuilder.scissor.offset = { 0, 0 };
	pipelineBuilder.scissor.extent = windowExtent;
	pipelineBuilder.rasterizer = VkInit::rasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);
	pipelineBuilder.multisampling = VkInit::multisamplingStateCreateInfo(msaaSamples);
	pipelineBuilder.colorBlendAttachment = VkInit::colorBlendAttachmentState();
	pipelineBuilder.depthStencil = VkInit::depthStencilCreateInfo(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

	VertexInputDescription vertexDescription = Vertex::getVertexDescription();
	pipelineBuilder.vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
	pipelineBuilder.vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();
	pipelineBuilder.vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
	pipelineBuilder.vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();

	// pbr pipeline
	VkPipelineLayoutCreateInfo pbrPipelineLayoutInfo = VkInit::pipelineLayoutCreateInfo();
	pbrPipelineLayoutInfo.pPushConstantRanges = &pushConstants;
	pbrPipelineLayoutInfo.pushConstantRangeCount = 1;

	VkDescriptorSetLayout pbrSetLayouts[] = { globalSetLayout, objectSetLayout, PBRSetLayout };
	pbrPipelineLayoutInfo.setLayoutCount = std::size(pbrSetLayouts);
	pbrPipelineLayoutInfo.pSetLayouts = pbrSetLayouts;
	VkPipelineLayout pbrPipelineLayout;
	VK_CHECK(vkCreatePipelineLayout(device, &pbrPipelineLayoutInfo, nullptr, &pbrPipelineLayout));

	pipelineBuilder.shaderStages.clear();
	pipelineBuilder.shaderStages.push_back(
		VkInit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, pbrVertShader));
	pipelineBuilder.shaderStages.push_back(
		VkInit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, pbrFragShader));
	pipelineBuilder.pipelineLayout = pbrPipelineLayout;
	VkPipeline PBRPipeline = pipelineBuilder.buildPipeline(device, renderPass);
	createMaterial(PBRPipeline, pbrPipelineLayout, "pbr");

	// shadowmap pipeline
	VkPipelineLayoutCreateInfo shadowMapPipelineLayoutInfo = VkInit::pipelineLayoutCreateInfo();
	shadowMapPipelineLayoutInfo.pPushConstantRanges = &pushConstants;
	shadowMapPipelineLayoutInfo.pushConstantRangeCount = 1;
	VkDescriptorSetLayout shadowMapSetLayouts[] = { globalSetLayout, objectSetLayout };
	shadowMapPipelineLayoutInfo.setLayoutCount = std::size(shadowMapSetLayouts);
	shadowMapPipelineLayoutInfo.pSetLayouts = shadowMapSetLayouts;
	VK_CHECK(vkCreatePipelineLayout(device, &shadowMapPipelineLayoutInfo, nullptr, &shadowMapPipelineLayout));

	pipelineBuilder.shaderStages.clear();
	pipelineBuilder.shaderStages.push_back(
		VkInit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, shadowMapVertShader));
	pipelineBuilder.shaderStages.push_back(
		VkInit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, shadowMapFragShader));
	pipelineBuilder.pipelineLayout = shadowMapPipelineLayout;
	pipelineBuilder.multisampling = VkInit::multisamplingStateCreateInfo(VK_SAMPLE_COUNT_1_BIT);
	pipelineBuilder.colorBlendAttachment.colorWriteMask = 0xf;
	pipelineBuilder.rasterizer.depthBiasEnable = VK_TRUE;
	pipelineBuilder.rasterizer.cullMode = VK_CULL_MODE_NONE;
	shadowMapPipeline = pipelineBuilder.buildPipeline(device, shadowPass);

	// Deletion of shader modules and pipelines
	vkDestroyShaderModule(device, pbrVertShader, nullptr);
	vkDestroyShaderModule(device, pbrFragShader, nullptr);
	vkDestroyShaderModule(device, shadowMapVertShader, nullptr);
	vkDestroyShaderModule(device, shadowMapFragShader, nullptr);

	mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(device, PBRPipeline, nullptr);
		vkDestroyPipelineLayout(device, pbrPipelineLayout, nullptr);
		vkDestroyPipeline(device, shadowMapPipeline, nullptr);
		vkDestroyPipelineLayout(device, shadowMapPipelineLayout, nullptr);
		});
}

void App::initScene()
{
	RenderObject background;
	background.mesh = getMesh("background");
	background.material = getMaterial("pbr");
	background.transformMatrix = glm::mat4{ 1.0f };
	renderables.push_back(background);

	RenderObject sphere;
	sphere.mesh = getMesh("sphere");
	sphere.material = getMaterial("pbr");
	sphere.transformMatrix = glm::mat4{ 1.0f };
	renderables.push_back(sphere);

	Material* pbrMat = getMaterial("pbr");

	// create sampler for texture
	VkSamplerCreateInfo samplerInfo = VkInit::samplerCreateInfo(VK_FILTER_LINEAR);
	VkSampler linearSampler;
	vkCreateSampler(device, &samplerInfo, nullptr, &linearSampler);

	// pbr stuff
	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.pNext = nullptr;
	allocInfo.descriptorPool = descriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &PBRSetLayout;
	vkAllocateDescriptorSets(device, &allocInfo, &pbrMat->textureSet);

	VkDescriptorImageInfo albedoImageInfo = {};
	albedoImageInfo.sampler = linearSampler;
	albedoImageInfo.imageView = loadedPBRMaterials["rustedIron"].albedo.imageView;
	albedoImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	VkWriteDescriptorSet albedo = VkInit::writeDescriptorImage(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, pbrMat->textureSet, &albedoImageInfo, 0);

	VkDescriptorImageInfo metallicImageInfo = {};
	metallicImageInfo.sampler = linearSampler;
	metallicImageInfo.imageView = loadedPBRMaterials["rustedIron"].metallic.imageView;
	metallicImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	VkWriteDescriptorSet metallic = VkInit::writeDescriptorImage(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, pbrMat->textureSet, &metallicImageInfo, 1);

	VkDescriptorImageInfo roughnessImageInfo = {};
	roughnessImageInfo.sampler = linearSampler;
	roughnessImageInfo.imageView = loadedPBRMaterials["rustedIron"].roughness.imageView;
	roughnessImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	VkWriteDescriptorSet roughness = VkInit::writeDescriptorImage(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, pbrMat->textureSet, &roughnessImageInfo, 2);

	VkDescriptorImageInfo normalImageInfo = {};
	normalImageInfo.sampler = linearSampler;
	normalImageInfo.imageView = loadedPBRMaterials["rustedIron"].normal.imageView;
	normalImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	VkWriteDescriptorSet normal = VkInit::writeDescriptorImage(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, pbrMat->textureSet, &normalImageInfo, 3);

	VkDescriptorImageInfo aoImageInfo = {};
	aoImageInfo.sampler = linearSampler;
	aoImageInfo.imageView = loadedPBRMaterials["rustedIron"].ao.imageView;
	aoImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	VkWriteDescriptorSet ao = VkInit::writeDescriptorImage(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, pbrMat->textureSet, &aoImageInfo, 4);

	VkWriteDescriptorSet setWrites[] = { albedo, metallic, roughness, normal, ao };
	vkUpdateDescriptorSets(device, 5, setWrites, 0, nullptr);

	mainDeletionQueue.push_function([=]()
		{
			vkDestroySampler(device, linearSampler, nullptr);
		});
}

void App::draw()
{
	if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED)
	{
		return;
	}
	// Wait until GPU has finished rendering last frame
	VK_CHECK(vkWaitForFences(device, 1, &getCurrentFrame().renderFence, true, 1000000000));
	VK_CHECK(vkResetFences(device, 1, &getCurrentFrame().renderFence));

	// Reset command buffer
	VK_CHECK(vkResetCommandBuffer(getCurrentFrame().mainCommandBuffer, 0));

	// Request image from the swapchain
	uint32_t swapchainImageIndex;
	VK_CHECK(vkAcquireNextImageKHR(device, swapchain, 1000000000, getCurrentFrame().presentSemaphore, nullptr, &swapchainImageIndex));

	// Get handle to current frame commandbuffer
	VkCommandBuffer cmd = getCurrentFrame().mainCommandBuffer;

	// Begin command buffer recording
	VkCommandBufferBeginInfo cmdBeginInfo = VkInit::commandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	// shadow mapping pass
	{
		for (size_t i = 0; i < 6; i++)
		{
			updateCubeFace(i, cmd, windowExtent);
		}
	}

	// Clear values
	// color of the screen (background)
	VkClearValue clearValue;
	clearValue.color = { 0.0f, 0.0f, 0.0f, 1.0f };
	// depth
	VkClearValue depthClear;
	depthClear.depthStencil.depth = 1.f;
	VkClearValue clearValues[] = { clearValue, depthClear };

	// Starting the renderpass
	VkRenderPassBeginInfo renderPassInfo = VkInit::renderpassBeginInfo(renderPass, windowExtent, frameBuffers[swapchainImageIndex]);
	renderPassInfo.clearValueCount = 2;
	renderPassInfo.pClearValues = &clearValues[0];
	vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	/* ----- RENDERING COMMANDS BEGIN ----- */
	
	drawObjects(cmd, renderables.data(), renderables.size(), false);

	/* ----- RENDERING COMMANDS END ----- */

	// Finalize render pass and command buffer
	vkCmdEndRenderPass(cmd);
	VK_CHECK(vkEndCommandBuffer(cmd));

	// submit image to the queue
	VkSubmitInfo submit = VkInit::submitInfo(&cmd);
	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	submit.pWaitDstStageMask = &waitStage;
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &getCurrentFrame().presentSemaphore;
	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &getCurrentFrame().renderSemaphore;
	VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submit, getCurrentFrame().renderFence));

	// put rendered image to visible window
	VkPresentInfoKHR presentInfo = VkInit::presentInfo();
	presentInfo.pSwapchains = &swapchain;
	presentInfo.swapchainCount = 1;
	presentInfo.pWaitSemaphores = &getCurrentFrame().renderSemaphore;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pImageIndices = &swapchainImageIndex;
	VK_CHECK(vkQueuePresentKHR(graphicsQueue, &presentInfo));

	frameNumber++;
}

FrameData& App::getCurrentFrame()
{
	return frames[frameNumber % FRAME_OVERLAP];
}

bool App::loadShaderModule(const char* filePath, VkShaderModule* outShaderModule)
{
	// Open the file with cursor at the end
	std::ifstream file(filePath, std::ios::ate | std::ios::binary);
	if (!file.is_open())
	{
		return false;
	}

	// Get filesize in bytes
	size_t fileSize = (size_t)file.tellg();

	// Read file content
	std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
	file.seekg(0);
	file.read((char*)buffer.data(), fileSize);
	file.close();

	VkShaderModuleCreateInfo shaderModuleInfo = {};
	shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shaderModuleInfo.pNext = nullptr;
	shaderModuleInfo.codeSize = buffer.size() * sizeof(uint32_t);
	shaderModuleInfo.pCode = buffer.data();

	VkShaderModule shaderModule;
	if (vkCreateShaderModule(device, &shaderModuleInfo, nullptr, &shaderModule) != VK_SUCCESS)
	{
		return false;
	}
	*outShaderModule = shaderModule;
	return true;
}

void App::loadMeshes()
{
	Mesh room{};
	room.loadFromObj("../../assets/viking_room.obj");
	uploadMesh(room);
	meshes["room"] = room;

	Mesh sphere{};
	sphere.loadFromObj("../../assets/sphere_smooth.obj");
	uploadMesh(sphere);
	meshes["sphere"] = sphere;

	Mesh background{};
	background.loadFromObj("../../assets/background.obj");
	uploadMesh(background);
	meshes["background"] = background;
}

void App::loadImages()
{
	PBRMaterial rustedIron = {};

	utils::loadImageFromFile(this, "../../assets/rustediron2_basecolor.png", rustedIron.albedo.image, VK_FORMAT_R8G8B8A8_SRGB);
	VkImageViewCreateInfo albedoInfo = VkInit::imageviewCreateInfo(VK_FORMAT_R8G8B8A8_SRGB, rustedIron.albedo.image.image, VK_IMAGE_ASPECT_COLOR_BIT);
	vkCreateImageView(device, &albedoInfo, nullptr, &rustedIron.albedo.imageView);

	utils::loadImageFromFile(this, "../../assets/rustediron2_metallic.png", rustedIron.metallic.image, VK_FORMAT_R8G8B8A8_SRGB);
	VkImageViewCreateInfo metallicInfo = VkInit::imageviewCreateInfo(VK_FORMAT_R8G8B8A8_SRGB, rustedIron.metallic.image.image, VK_IMAGE_ASPECT_COLOR_BIT);
	vkCreateImageView(device, &metallicInfo, nullptr, &rustedIron.metallic.imageView);

	utils::loadImageFromFile(this, "../../assets/rustediron2_roughness.png", rustedIron.roughness.image, VK_FORMAT_R8G8B8A8_SRGB);
	VkImageViewCreateInfo roughnessInfo = VkInit::imageviewCreateInfo(VK_FORMAT_R8G8B8A8_SRGB, rustedIron.roughness.image.image, VK_IMAGE_ASPECT_COLOR_BIT);
	vkCreateImageView(device, &roughnessInfo, nullptr, &rustedIron.roughness.imageView);

	utils::loadImageFromFile(this, "../../assets/rustediron2_normal.png", rustedIron.normal.image, VK_FORMAT_R8G8B8A8_UNORM);
	VkImageViewCreateInfo normalInfo = VkInit::imageviewCreateInfo(VK_FORMAT_R8G8B8A8_UNORM, rustedIron.normal.image.image, VK_IMAGE_ASPECT_COLOR_BIT);
	vkCreateImageView(device, &normalInfo, nullptr, &rustedIron.normal.imageView);

	utils::loadImageFromFile(this, "../../assets/rustediron2_ao.png", rustedIron.ao.image, VK_FORMAT_R8G8B8A8_SRGB);
	VkImageViewCreateInfo aoInfo = VkInit::imageviewCreateInfo(VK_FORMAT_R8G8B8A8_SRGB, rustedIron.ao.image.image, VK_IMAGE_ASPECT_COLOR_BIT);
	vkCreateImageView(device, &aoInfo, nullptr, &rustedIron.ao.imageView);

	loadedPBRMaterials["rustedIron"] = rustedIron;

	mainDeletionQueue.push_function([=]()
		{
			vkDestroyImageView(device, rustedIron.albedo.imageView, nullptr);
			vkDestroyImageView(device, rustedIron.metallic.imageView, nullptr);
			vkDestroyImageView(device, rustedIron.roughness.imageView, nullptr);
			vkDestroyImageView(device, rustedIron.normal.imageView, nullptr);
			vkDestroyImageView(device, rustedIron.ao.imageView, nullptr);
		});
}

void App::uploadMesh(Mesh& mesh)
{
	// get buffer sizes
	const size_t vertexBufferSize = mesh.vertices.size() * sizeof(Vertex);
	const size_t indexBufferSize = mesh.indices.size() * sizeof(uint32_t);
	// create staging buffers for CPU-side
	VkBufferCreateInfo stagingVertexBufferInfo = {};
	stagingVertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	stagingVertexBufferInfo.pNext = nullptr;
	stagingVertexBufferInfo.size = vertexBufferSize;
	stagingVertexBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	VkBufferCreateInfo stagingIndexBufferInfo = {};
	stagingIndexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	stagingIndexBufferInfo.pNext = nullptr;
	stagingIndexBufferInfo.size = indexBufferSize;
	stagingIndexBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	VmaAllocationCreateInfo vmaAllocInfo = {};
	vmaAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	// allocate buffers
	AllocatedBuffer stagingVertexBuffer;
	VK_CHECK(vmaCreateBuffer(allocator, &stagingVertexBufferInfo, &vmaAllocInfo,
		&stagingVertexBuffer.buffer, &stagingVertexBuffer.allocation, nullptr));

	AllocatedBuffer stagingIndexBuffer;
	VK_CHECK(vmaCreateBuffer(allocator, &stagingIndexBufferInfo, &vmaAllocInfo,
		&stagingIndexBuffer.buffer, &stagingIndexBuffer.allocation, nullptr));

	// copy mesh data
	void* vData;
	vmaMapMemory(allocator, stagingVertexBuffer.allocation, &vData);
	memcpy(vData, mesh.vertices.data(), vertexBufferSize);
	vmaUnmapMemory(allocator, stagingVertexBuffer.allocation);

	void* iData;
	vmaMapMemory(allocator, stagingIndexBuffer.allocation, &iData);
	memcpy(iData, mesh.indices.data(), indexBufferSize);
	vmaUnmapMemory(allocator, stagingIndexBuffer.allocation);

	// create GPU-side buffers
	VkBufferCreateInfo vertexBufferInfo = {};
	vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vertexBufferInfo.pNext = nullptr;
	vertexBufferInfo.size = vertexBufferSize;
	vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	VkBufferCreateInfo indexBufferInfo = {};
	indexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	indexBufferInfo.pNext = nullptr;
	indexBufferInfo.size = indexBufferSize;
	indexBufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	vmaAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	// allocate the buffers
	VK_CHECK(vmaCreateBuffer(allocator, &vertexBufferInfo, &vmaAllocInfo,
		&mesh.vertexBuffer.buffer, &mesh.vertexBuffer.allocation, nullptr));
	VK_CHECK(vmaCreateBuffer(allocator, &indexBufferInfo, &vmaAllocInfo,
		&mesh.indexBuffer.buffer, &mesh.indexBuffer.allocation, nullptr));

	// copy buffer contents
	immediateSubmit([=](VkCommandBuffer cmd)
	{
		VkBufferCopy copyVertex;
		copyVertex.dstOffset = 0;
		copyVertex.srcOffset = 0;
		copyVertex.size = vertexBufferSize;
		vkCmdCopyBuffer(cmd, stagingVertexBuffer.buffer, mesh.vertexBuffer.buffer, 1, &copyVertex);

		VkBufferCopy copyIndex;
		copyIndex.dstOffset = 0;
		copyIndex.srcOffset = 0;
		copyIndex.size = indexBufferSize;
		vkCmdCopyBuffer(cmd, stagingIndexBuffer.buffer, mesh.indexBuffer.buffer, 1, &copyIndex);
	});

	// cleanup
	mainDeletionQueue.push_function([=]()
		{
			vmaDestroyBuffer(allocator, mesh.vertexBuffer.buffer, mesh.vertexBuffer.allocation);
			vmaDestroyBuffer(allocator, mesh.indexBuffer.buffer, mesh.indexBuffer.allocation);
		});
	vmaDestroyBuffer(allocator, stagingVertexBuffer.buffer, stagingVertexBuffer.allocation);
	vmaDestroyBuffer(allocator, stagingIndexBuffer.buffer, stagingIndexBuffer.allocation);
}

Material* App::createMaterial(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name)
{
	Material material;
	material.pipeline = pipeline;
	material.pipelineLayout = layout;
	materials[name] = material;
	return &materials[name];
}

Material* App::getMaterial(const std::string& name)
{
	auto it = materials.find(name);
	if (it == materials.end())
	{
		return nullptr;
	}
	else
	{
		return &(*it).second;
	}
}

Mesh* App::getMesh(const std::string& name)
{
	auto it = meshes.find(name);
	if (it == meshes.end())
	{
		return nullptr;
	}
	else
	{
		return &(*it).second;
	}
}

void App::drawObjects(VkCommandBuffer commandBuffer, RenderObject* first, int count, bool isShadowPass)
{
	glm::vec3 camPos = { 0.f, 0.f, -5.f };
	glm::mat4 view = glm::translate(glm::mat4(1.f), camPos);
	glm::mat4 projection = glm::perspective(glm::radians(60.f), (float) windowExtent.width / windowExtent.height, 1.0f, 64.0f);
	projection[1][1] *= -1;

	// send camera data to uniform buffer
	GPUCameraData camData;
	camData.proj = projection;
	camData.view = view;
	camData.viewproj = projection * view;
	void* data;
	vmaMapMemory(allocator, getCurrentFrame().cameraBuffer.allocation, &data);
	memcpy(data, &camData, sizeof(GPUCameraData));
	vmaUnmapMemory(allocator, getCurrentFrame().cameraBuffer.allocation);

	// allocating scene parameters
	float d = (frameNumber / 144.f);
	sceneParameters.ambientColor = { 0.03f, 0.03f , 0.03f, 1.f };
	sceneParameters.lightPosition = { 2.f*sin(d), 0.f, 2.f*cos(d), 1.f };
	sceneParameters.lightColor = { 150.f, 150.f, 150.f, 1.f };
	char* sceneData;
	vmaMapMemory(allocator, sceneParameterBuffer.allocation, (void**)&sceneData);
	int frameIndex = frameNumber % FRAME_OVERLAP;
	sceneData += padUniformBufferSize(sizeof(GPUSceneData)) * frameIndex;
	memcpy(sceneData, &sceneParameters, sizeof(GPUSceneData));
	vmaUnmapMemory(allocator, sceneParameterBuffer.allocation);

	// send light position data for shadowmapping
	lightUBO.projection = glm::perspective(glm::radians(90.f), 1.f, 0.1f, 32.f);
	lightUBO.projection[1][1] *= -1;
	lightUBO.model = glm::translate(glm::mat4(1.0f), glm::vec3(-sceneParameters.lightPosition.x, -sceneParameters.lightPosition.y, -sceneParameters.lightPosition.z));
	void* lightMVPData;
	vmaMapMemory(allocator, getCurrentFrame().lightMVPBuffer.allocation, &lightMVPData);
	memcpy(lightMVPData, &lightUBO, sizeof(GPUlightMVPData));
	vmaUnmapMemory(allocator, getCurrentFrame().lightMVPBuffer.allocation);

	// allocating object buffer
	void* objectData;
	vmaMapMemory(allocator, getCurrentFrame().objectBuffer.allocation, &objectData);
	GPUObjectData* objectSSBO = (GPUObjectData*)objectData;
	for (int i = 0; i < count; i++)
	{
		RenderObject& object = first[i];
		objectSSBO[i].modelMatrix = object.transformMatrix;
		// rotate
		//objectSSBO[i].modelMatrix = glm::rotate(objectSSBO[i].modelMatrix, glm::radians(frameNumber * 0.10f), glm::vec3(0, 1, 0));
	}
	vmaUnmapMemory(allocator, getCurrentFrame().objectBuffer.allocation);

	Mesh* lastMesh = nullptr;
	Material* lastMaterial = nullptr;
	for (int i = 0; i < count; i++)
	{
		RenderObject& object = first[i];

		if (object.material != lastMaterial)
		{
			VkPipeline* pipeline;
			VkPipelineLayout* pipelineLayout;
			if (isShadowPass)
			{
				pipeline = &shadowMapPipeline;
				pipelineLayout = &shadowMapPipelineLayout;
			}
			else
			{
				pipeline = &object.material->pipeline;
				pipelineLayout = &object.material->pipelineLayout;
			}
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline);
			lastMaterial = object.material;
			// camera data descriptor
			uint32_t uniformOffset = padUniformBufferSize(sizeof(GPUSceneData)) * frameIndex;
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, *pipelineLayout,
				0, 1, &getCurrentFrame().globalDescriptor, 1, &uniformOffset);
			// object data descriptor
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, *pipelineLayout,
				1, 1, &getCurrentFrame().objectDescriptor, 0, nullptr);
			if (object.material->textureSet != VK_NULL_HANDLE && !isShadowPass)
			{
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, *pipelineLayout,
					2, 1, &object.material->textureSet, 0, nullptr);
			}
		}

		glm::mat4 model = object.transformMatrix;
		model = glm::rotate(model, glm::radians(frameNumber * 0.5f), glm::vec3(0, 1, 0));

		MeshPushConstants constants;
		constants.cameraPosition = camPos;

		vkCmdPushConstants(commandBuffer, object.material->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants), &constants);

		if (object.mesh != lastMesh)
		{
			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, &object.mesh->vertexBuffer.buffer, &offset);
			vkCmdBindIndexBuffer(commandBuffer, object.mesh->indexBuffer.buffer, offset, VK_INDEX_TYPE_UINT32);
			lastMesh = object.mesh;
		}
		vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(object.mesh->indices.size()), 1, 0, 0, i);
	}
}

AllocatedBuffer App::createBuffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.pNext = nullptr;
	bufferInfo.size = allocSize;
	bufferInfo.usage = usage;
	VmaAllocationCreateInfo vmaAllocInfo = {};
	vmaAllocInfo.usage = memoryUsage;

	AllocatedBuffer newBuffer;
	VK_CHECK(vmaCreateBuffer(allocator, &bufferInfo, &vmaAllocInfo,
		&newBuffer.buffer,
		&newBuffer.allocation,
		nullptr));
	return newBuffer;
}

void App::initDescriptors()
{
	// descriptor pool
	std::vector<VkDescriptorPoolSize> sizes =
	{
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10},
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10},
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10}
	};

	// descriptor pool creation
	VkDescriptorPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.pNext = nullptr;
	poolInfo.flags = 0;
	poolInfo.maxSets = 10;
	poolInfo.poolSizeCount = (uint32_t)sizes.size();
	poolInfo.pPoolSizes = sizes.data();
	vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);

	// cam binding
	VkDescriptorSetLayoutBinding camBinding = VkInit::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
	// scene data binding
	VkDescriptorSetLayoutBinding sceneBinding = VkInit::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1);
	// shadowmap binding
	VkDescriptorSetLayoutBinding shadowMapBinding = VkInit::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2);
	// lightMVP binding
	VkDescriptorSetLayoutBinding lightMVPBinding = VkInit::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 3);

	VkDescriptorSetLayoutBinding bindings[] = { camBinding, sceneBinding, shadowMapBinding, lightMVPBinding };

	VkDescriptorSetLayoutCreateInfo setinfo = {};
	setinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	setinfo.pNext = nullptr;
	setinfo.bindingCount = std::size(bindings);
	setinfo.flags = 0;
	setinfo.pBindings = bindings;
	vkCreateDescriptorSetLayout(device, &setinfo, nullptr, &globalSetLayout);

	// object binding
	VkDescriptorSetLayoutBinding objectBinding = VkInit::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
	VkDescriptorSetLayoutCreateInfo set2info = {};
	set2info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	set2info.pNext = nullptr;
	set2info.bindingCount = 1;
	set2info.flags = 0;
	set2info.pBindings = &objectBinding;
	vkCreateDescriptorSetLayout(device, &set2info, nullptr, &objectSetLayout);

	// pbr binding
	VkDescriptorSetLayoutBinding albedoBinding = VkInit::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	VkDescriptorSetLayoutBinding metallicBinding = VkInit::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1);
	VkDescriptorSetLayoutBinding roughnessBinding = VkInit::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2);
	VkDescriptorSetLayoutBinding normalBinding = VkInit::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3);
	VkDescriptorSetLayoutBinding aoBinding = VkInit::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 4);
	VkDescriptorSetLayoutBinding pbrBindings[] = { albedoBinding, metallicBinding, roughnessBinding, normalBinding, aoBinding };
	VkDescriptorSetLayoutCreateInfo set4info = {};
	set4info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	set4info.pNext = nullptr;
	set4info.bindingCount = std::size(pbrBindings);
	set4info.flags = 0;
	set4info.pBindings = pbrBindings;
	vkCreateDescriptorSetLayout(device, &set4info, nullptr, &PBRSetLayout);

	const size_t sceneParameterBufferSize = FRAME_OVERLAP * padUniformBufferSize(sizeof(GPUSceneData));
	sceneParameterBuffer = createBuffer(sceneParameterBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	for (int i = 0; i < FRAME_OVERLAP; i++)
	{
		frames[i].cameraBuffer = createBuffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		
		const int MAX_OBJECTS = 10000;
		frames[i].objectBuffer = createBuffer(sizeof(GPUObjectData) * MAX_OBJECTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

		frames[i].lightMVPBuffer = createBuffer(sizeof(GPUlightMVPData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

		// alloc one descritpor set for each frame
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.pNext = nullptr;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &globalSetLayout;
		vkAllocateDescriptorSets(device, &allocInfo, &frames[i].globalDescriptor);

		// allocate descriptor set for object buffer
		VkDescriptorSetAllocateInfo objectSetAlloc = {};
		objectSetAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		objectSetAlloc.pNext = nullptr;
		objectSetAlloc.descriptorPool = descriptorPool;
		objectSetAlloc.descriptorSetCount = 1;
		objectSetAlloc.pSetLayouts = &objectSetLayout;
		vkAllocateDescriptorSets(device, &objectSetAlloc, &frames[i].objectDescriptor);

		// info about camera buffer for descriptor
		VkDescriptorBufferInfo cameraInfo = {};
		cameraInfo.buffer = frames[i].cameraBuffer.buffer;
		cameraInfo.offset = 0;
		cameraInfo.range = sizeof(GPUCameraData);

		// info about scene buffer for descriptor
		VkDescriptorBufferInfo sceneInfo = {};
		sceneInfo.buffer = sceneParameterBuffer.buffer;
		sceneInfo.offset = 0;
		sceneInfo.range = sizeof(GPUSceneData);

		// info about shadow map
		VkDescriptorImageInfo shadowMapInfo = {};
		shadowMapInfo.sampler = shadowCubeSampler;
		shadowMapInfo.imageView = shadowCubeImageView;
		shadowMapInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		// info about lightMVP buffer
		VkDescriptorBufferInfo lightMVPInfo = {};
		lightMVPInfo.buffer = frames[i].lightMVPBuffer.buffer;
		lightMVPInfo.offset = 0;
		lightMVPInfo.range = sizeof(GPUlightMVPData);

		// info about object buffer for descriptor
		VkDescriptorBufferInfo objectBufferInfo = {};
		objectBufferInfo.buffer = frames[i].objectBuffer.buffer;
		objectBufferInfo.offset = 0;
		objectBufferInfo.range = sizeof(GPUObjectData) * MAX_OBJECTS;

		// write descriptors
		VkWriteDescriptorSet cameraWrite = VkInit::writeDescriptorBuffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, frames[i].globalDescriptor, &cameraInfo, 0);
		VkWriteDescriptorSet sceneWrite = VkInit::writeDescriptorBuffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, frames[i].globalDescriptor, &sceneInfo, 1);
		VkWriteDescriptorSet shadowMapWrite = VkInit::writeDescriptorImage(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, frames[i].globalDescriptor, &shadowMapInfo, 2);
		VkWriteDescriptorSet lightMVPWrite = VkInit::writeDescriptorBuffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, frames[i].globalDescriptor, &lightMVPInfo, 3);
		VkWriteDescriptorSet objectWrite = VkInit::writeDescriptorBuffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, frames[i].objectDescriptor, &objectBufferInfo, 0);

		VkWriteDescriptorSet setWrites[] = { cameraWrite, sceneWrite,shadowMapWrite, lightMVPWrite, objectWrite };

		vkUpdateDescriptorSets(device, 5, setWrites, 0, nullptr);
	}

	// add buffers to deletion queues
	mainDeletionQueue.push_function([&]()
		{
			vmaDestroyBuffer(allocator, sceneParameterBuffer.buffer, sceneParameterBuffer.allocation);
			vkDestroyDescriptorSetLayout(device, objectSetLayout, nullptr);
			vkDestroyDescriptorSetLayout(device, globalSetLayout, nullptr);
			vkDestroyDescriptorSetLayout(device, PBRSetLayout, nullptr);
			vkDestroyDescriptorPool(device, descriptorPool, nullptr);

			for (int i = 0; i < FRAME_OVERLAP; i++)
			{
				vmaDestroyBuffer(allocator, frames[i].cameraBuffer.buffer, frames[i].cameraBuffer.allocation);
				vmaDestroyBuffer(allocator, frames[i].objectBuffer.buffer, frames[i].objectBuffer.allocation);
				vmaDestroyBuffer(allocator, frames[i].lightMVPBuffer.buffer, frames[i].lightMVPBuffer.allocation);
			}
		});
}

void App::immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function)
{
	VkCommandBuffer cmd = uploadContext.commandBuffer;
	// begin commandbuffer recording
	VkCommandBufferBeginInfo cmdBeginInfo = VkInit::commandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
	function(cmd);
	VK_CHECK(vkEndCommandBuffer(cmd));
	
	// submit command buffer and execute it
	VkSubmitInfo submit = VkInit::submitInfo(&cmd);
	VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submit, uploadContext.uploadFence));
	vkWaitForFences(device, 1, &uploadContext.uploadFence, true, 9999999999);
	vkResetFences(device, 1, &uploadContext.uploadFence);

	// reset the command buffers
	vkResetCommandPool(device, uploadContext.commandPool, 0);
}

size_t App::padUniformBufferSize(size_t originalSize)
{
	// from https://github.com/SaschaWillems/Vulkan/tree/master/examples/dynamicuniformbuffer
	size_t minAlignment = gpuProperties.limits.minUniformBufferOffsetAlignment;
	size_t alignedSize = originalSize;
	if (minAlignment > 0)
	{
		alignedSize = (alignedSize + minAlignment - 1) & ~(minAlignment - 1);
	}
	return alignedSize;
}

bool App::isFormatFilterable(VkPhysicalDevice physDevice, VkFormat format, VkImageTiling tiling)
{
	VkFormatProperties formatProps;
	vkGetPhysicalDeviceFormatProperties(physDevice, format, &formatProps);
	if (tiling == VK_IMAGE_TILING_OPTIMAL)
		return formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
	if (tiling == VK_IMAGE_TILING_LINEAR)
		return formatProps.linearTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
	return false;
}

void App::initShadowPass()
{
	VkAttachmentDescription attachmentDescriptions[2] = {};

	attachmentDescriptions[0].format = VK_FORMAT_R32_SFLOAT;
	attachmentDescriptions[0].samples = VK_SAMPLE_COUNT_1_BIT;
	attachmentDescriptions[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachmentDescriptions[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachmentDescriptions[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachmentDescriptions[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachmentDescriptions[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachmentDescriptions[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	attachmentDescriptions[1].format = depthFormat;
	attachmentDescriptions[1].samples = VK_SAMPLE_COUNT_1_BIT;
	attachmentDescriptions[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachmentDescriptions[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachmentDescriptions[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachmentDescriptions[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachmentDescriptions[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachmentDescriptions[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

	VkAttachmentReference colorReference = {};
	colorReference.attachment = 0;
	colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthReference = {};
	depthReference.attachment = 1;
	depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorReference;
	subpass.pDepthStencilAttachment = &depthReference;

	// layout transitions done with subpass dependencies
	std::array<VkSubpassDependency, 2> dependencies;

	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0;
	dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
	dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	dependencies[1].srcSubpass = 0;
	dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
	dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	VkRenderPassCreateInfo renderPassCreateInfo = {};
	renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassCreateInfo.pNext = nullptr;
	renderPassCreateInfo.attachmentCount = 2;
	renderPassCreateInfo.pAttachments = attachmentDescriptions;
	renderPassCreateInfo.subpassCount = 1;
	renderPassCreateInfo.pSubpasses = &subpass;
	renderPassCreateInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
	renderPassCreateInfo.pDependencies = dependencies.data();

	VK_CHECK(vkCreateRenderPass(device, &renderPassCreateInfo, nullptr, &shadowPass));

	mainDeletionQueue.push_function([=]()
		{
			vkDestroyRenderPass(device, shadowPass, nullptr);
		});
}

void App::initShadowPassFramebuffer()
{
	// create image and imageview for shadow image framebuffer
	VkExtent3D imageExtent = {
		2048u,
		2048u,
		1
	};
	VkImageCreateInfo imageInfo = VkInit::imageCreateInfo(
		depthFormat,
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		imageExtent,
		VK_SAMPLE_COUNT_1_BIT
	);
	VmaAllocationCreateInfo imageAllocInfo = {};
	imageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	imageAllocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	vmaCreateImage(allocator, &imageInfo, &imageAllocInfo, &shadowImage.image, &shadowImage.allocation, nullptr);
	VkImageViewCreateInfo imageViewInfo = VkInit::imageviewCreateInfo(depthFormat, shadowImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);
	VK_CHECK(vkCreateImageView(device, &imageViewInfo, nullptr, &shadowImageView));

	// create sampler for shadow image
	VkFilter shadowmapFilter = isFormatFilterable(gpu, depthFormat, VK_IMAGE_TILING_OPTIMAL) ?
		VK_FILTER_LINEAR :
		VK_FILTER_NEAREST;
	VkSamplerCreateInfo samplerInfo = VkInit::samplerCreateInfo(shadowmapFilter, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &shadowSampler));

	// create framebuffer for shadow image
	VkImageView attachments[2];
	attachments[1] = shadowImageView;
	VkExtent2D fbExtent = { 2048u, 2048u };
	VkFramebufferCreateInfo fbInfo = VkInit::framebufferCreateInfo(shadowPass, fbExtent);
	fbInfo.attachmentCount = 2;
	fbInfo.pAttachments = attachments;
	fbInfo.layers = 1;
	for (size_t i = 0; i < 6; i++)
	{
		attachments[0] = shadowCubeFaceImageViews[i];
		VK_CHECK(vkCreateFramebuffer(device, &fbInfo, nullptr, &shadowPassCubeFrameBuffers[i]));
		mainDeletionQueue.push_function([=]()
			{
				vkDestroyFramebuffer(device, shadowPassCubeFrameBuffers[i], nullptr);
			});
	}


	mainDeletionQueue.push_function([=]()
		{
			vkDestroyImageView(device, shadowImageView, nullptr);
			vmaDestroyImage(allocator, shadowImage.image, shadowImage.allocation);
			vkDestroySampler(device, shadowSampler, nullptr);
		});
}

void App::initCubeMap()
{
	VkFormat format = VK_FORMAT_R32_SFLOAT;

	VkImageCreateInfo imageCreateInfo = VkInit::imageCreateInfo(format,
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		{ 2048, 2048, 1 },
		VK_SAMPLE_COUNT_1_BIT);
	imageCreateInfo.arrayLayers = 6;
	imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageCreateInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
	VmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	allocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	vmaCreateImage(allocator, &imageCreateInfo, &allocInfo, &shadowCubeImage.image, &shadowCubeImage.allocation, nullptr);

	VkSamplerCreateInfo sampler = VkInit::samplerCreateInfo(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
	sampler.mipLodBias = 0.0f;
	sampler.maxAnisotropy = 1.0f;
	sampler.compareOp = VK_COMPARE_OP_NEVER;
	sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	VK_CHECK(vkCreateSampler(device, &sampler, nullptr, &shadowCubeSampler));

	VkImageViewCreateInfo viewCI = VkInit::imageviewCreateInfo(format, VK_NULL_HANDLE, VK_IMAGE_ASPECT_COLOR_BIT);
	viewCI.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
	viewCI.components = { VK_COMPONENT_SWIZZLE_R };
	viewCI.subresourceRange.layerCount = 6;
	viewCI.image = shadowCubeImage.image;
	VK_CHECK(vkCreateImageView(device, &viewCI, nullptr, &shadowCubeImageView));

	viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewCI.subresourceRange.layerCount = 1;
	viewCI.image = shadowCubeImage.image;

	for (size_t i = 0; i < 6; i++)
	{
		viewCI.subresourceRange.baseArrayLayer = i;
		VK_CHECK(vkCreateImageView(device, &viewCI, nullptr, &shadowCubeFaceImageViews[i]));
		mainDeletionQueue.push_function([=]()
			{
				vkDestroyImageView(device, shadowCubeFaceImageViews[i], nullptr);
			});
	}

	mainDeletionQueue.push_function([=]()
		{
			vkDestroyImageView(device, shadowCubeImageView, nullptr);
			vmaDestroyImage(allocator, shadowCubeImage.image, shadowCubeImage.allocation);
		});
}

void App::updateCubeFace(uint32_t face, VkCommandBuffer cmd, VkExtent2D extent)
{
	VkClearValue clearValues[2];
	clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
	clearValues[1].depthStencil = { 1.0f, 0 };
	VkRenderPassBeginInfo renderPassBeginInfo = VkInit::renderpassBeginInfo(shadowPass, extent, shadowPassCubeFrameBuffers[face]);
	renderPassBeginInfo.clearValueCount = 2;
	renderPassBeginInfo.pClearValues = clearValues;

	lightUBO.view = glm::mat4(1.0f);
	switch (face)
	{
	case 0: // pos x
		lightUBO.view = glm::rotate(lightUBO.view, glm::radians(90.f), glm::vec3(0.0f, 1.0f, 0.0f));
		lightUBO.view = glm::rotate(lightUBO.view, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		break;
	case 1:	// neg x
		lightUBO.view = glm::rotate(lightUBO.view, glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		lightUBO.view = glm::rotate(lightUBO.view, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		break;
	case 2:	// pos y
		lightUBO.view = glm::rotate(lightUBO.view, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		break;
	case 3:	// neg y
		lightUBO.view = glm::rotate(lightUBO.view, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		break;
	case 4:	// pos z
		lightUBO.view = glm::rotate(lightUBO.view, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		break;
	case 5:	// neg z
		lightUBO.view = glm::rotate(lightUBO.view, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		break;
	}

	vkCmdBeginRenderPass(cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
	drawObjects(cmd, renderables.data(), renderables.size(), true);
	vkCmdEndRenderPass(cmd);
}
