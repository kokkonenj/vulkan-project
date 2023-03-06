#include "app.h"
#include <SDL.h>
#include <SDL_vulkan.h>
#include <VkBootstrap.h>
#include <iostream>
#include <fstream>
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

	VkImageCreateInfo depthImageInfo = VkInit::imageCreateInfo(depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImageExtent);

	// allocate depth image from GPU local memory
	VmaAllocationCreateInfo depthImageAllocInfo = {};
	depthImageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	depthImageAllocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	vmaCreateImage(allocator, &depthImageInfo, &depthImageAllocInfo, &depthImage.image, &depthImage.allocation, nullptr);

	// build image-view for depth image
	VkImageViewCreateInfo depthViewInfo = VkInit::imageviewCreateInfo(depthFormat, depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);
	VK_CHECK(vkCreateImageView(device, &depthViewInfo, nullptr, &depthImageView));

	// cleanup
	mainDeletionQueue.push_function([=]()
		{
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
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference colorAttatchmentReference = {};
	colorAttatchmentReference.attachment = 0;
	colorAttatchmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	// depth attachment
	VkAttachmentDescription depthAttachment = {};
	depthAttachment.flags = 0;
	depthAttachment.format = depthFormat;
	depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthAttachmentReference = {};
	depthAttachmentReference.attachment = 1;
	depthAttachmentReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	// subpass
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttatchmentReference;
	subpass.pDepthStencilAttachment = &depthAttachmentReference;

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
	VkAttachmentDescription attachments[2] = { colorAttachment, depthAttachment };

	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = 2;
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
		VkImageView attachments[2];
		attachments[0] = swapchainImageViews[i];
		attachments[1] = depthImageView;

		frameBufferInfo.pAttachments = attachments;
		frameBufferInfo.attachmentCount = 2;
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
	VkShaderModule triangleFragmentShader;
	if (!loadShaderModule("../../shaders/triangle.frag.spv", &triangleFragmentShader))
	{
		std::cout << "Error loading triangle fragment shader module" << std::endl;
	}
	else
	{
		std::cout << "Triangle fragment shader successfully loaded" << std::endl;
	}

	VkShaderModule triangleVertexShader;
	if (!loadShaderModule("../../shaders/triangle.vert.spv", &triangleVertexShader))
	{
		std::cout << "Error loading triangle vertex shader module" << std::endl;
	}
	else
	{
		std::cout << "Triangle vertex shader successfully loaded" << std::endl;
	}

	// Create-info for vertex and fragment stages
	PipelineBuilder pipelineBuilder;
	pipelineBuilder.shaderStages.push_back(
		VkInit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, triangleVertexShader));
	pipelineBuilder.shaderStages.push_back(
		VkInit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragmentShader));
	
	// Input info controls how to read vertices from buffers
	pipelineBuilder.vertexInputInfo = VkInit::vertexInputStateCreateInfo();

	// Input assembly controls how to draw (triangle list, lines or points)
	pipelineBuilder.inputAssembly = VkInit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	// Viewport and scissor from swapchain extents
	pipelineBuilder.viewport.x = 0.0f;
	pipelineBuilder.viewport.y = 0.0f;
	pipelineBuilder.viewport.width = (float)windowExtent.width;
	pipelineBuilder.viewport.height = (float)windowExtent.height;
	pipelineBuilder.viewport.minDepth = 0.0f;
	pipelineBuilder.viewport.maxDepth = 1.0f;
	pipelineBuilder.scissor.offset = { 0, 0 };
	pipelineBuilder.scissor.extent = windowExtent;

	// Configure rasterizer
	pipelineBuilder.rasterizer = VkInit::rasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);

	// Configure multisampling
	pipelineBuilder.multisampling = VkInit::multisamplingStateCreateInfo();

	// Configure color blend attachment (no blending, RGBA)
	pipelineBuilder.colorBlendAttachment = VkInit::colorBlendAttachmentState();

	// Default depth testing
	pipelineBuilder.depthStencil = VkInit::depthStencilCreateInfo(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

	// Build mesh pipeline
	VertexInputDescription vertexDescription = Vertex::getVertexDescription();
	pipelineBuilder.vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
	pipelineBuilder.vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();
	pipelineBuilder.vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
	pipelineBuilder.vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();
	pipelineBuilder.shaderStages.clear();

	VkShaderModule meshVertexShader;
	if (!loadShaderModule("../../shaders/mesh.vert.spv", &meshVertexShader))
	{
		std::cout << "Error loading mesh vertex shader module" << std::endl;
	}
	else
	{
		std::cout << "Mesh vertex shader successfully loaded" << std::endl;
	}

	pipelineBuilder.shaderStages.push_back(
		VkInit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, meshVertexShader));
	pipelineBuilder.shaderStages.push_back(
		VkInit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragmentShader));

	// Mesh pipeline layout definition
	VkPipelineLayoutCreateInfo meshPipelineLayoutInfo = VkInit::pipelineLayoutCreateInfo();
	VkPushConstantRange pushConstant;
	pushConstant.offset = 0;
	pushConstant.size = sizeof(MeshPushConstants);
	pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	meshPipelineLayoutInfo.pPushConstantRanges = &pushConstant;
	meshPipelineLayoutInfo.pushConstantRangeCount = 1;
	// global & object set layout
	VkDescriptorSetLayout setLayouts[] = { globalSetLayout, objectSetLayout };
	meshPipelineLayoutInfo.setLayoutCount = 2;
	meshPipelineLayoutInfo.pSetLayouts = setLayouts;
	VK_CHECK(vkCreatePipelineLayout(device, &meshPipelineLayoutInfo, nullptr, &meshPipelineLayout));

	// build default mesh pipeline
	pipelineBuilder.pipelineLayout = meshPipelineLayout;
	meshPipeline = pipelineBuilder.buildPipeline(device, renderPass);
	createMaterial(meshPipeline, meshPipelineLayout, "defaultMesh");

	// Deletion of shader modules and pipelines
	vkDestroyShaderModule(device, meshVertexShader, nullptr);
	vkDestroyShaderModule(device, triangleFragmentShader, nullptr);
	vkDestroyShaderModule(device, triangleVertexShader, nullptr);
	mainDeletionQueue.push_function([=]()
		{
			vkDestroyPipeline(device, meshPipeline, nullptr);
			vkDestroyPipelineLayout(device, meshPipelineLayout, nullptr);
		});
}

void App::initScene()
{
	RenderObject monkey;
	monkey.mesh = getMesh("monkey");
	monkey.material = getMaterial("defaultMesh");
	monkey.transformMatrix = glm::mat4{ 1.0f };
	renderables.push_back(monkey);
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
	
	drawObjects(cmd, renderables.data(), renderables.size());

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
	Mesh triangleMesh{};
	triangleMesh.vertices.resize(3);

	triangleMesh.vertices[0].position = { 0.5f, 0.5f, 0.0f };
	triangleMesh.vertices[1].position = { -0.5f, 0.5f, 0.0f };
	triangleMesh.vertices[2].position = { 0.0f, -0.5f, 0.0f };

	triangleMesh.vertices[0].color = { 0.0f, 0.0f, 1.0f };
	triangleMesh.vertices[1].color = { 0.0f, 1.0f, 0.0f };
	triangleMesh.vertices[2].color = { 1.0f, 0.0f, 0.0f };

	triangleMesh.indices = { 0, 1, 2 };

	Mesh monkeyMesh{};
	monkeyMesh.loadFromObj("../../assets/monkey_smooth.obj");

	uploadMesh(triangleMesh);
	uploadMesh(monkeyMesh);
	
	meshes["monkey"] = monkeyMesh;
	meshes["triangle"] = triangleMesh;
}

void App::loadImages()
{
	Texture lostEmpire;
	utils::loadImageFromFile(this, "../../assets/lost_empire-RGBA.png", lostEmpire.image);
	
	VkImageViewCreateInfo imageInfo = VkInit::imageviewCreateInfo(VK_FORMAT_R8G8B8A8_SRGB, lostEmpire.image.image, VK_IMAGE_ASPECT_COLOR_BIT);
	vkCreateImageView(device, &imageInfo, nullptr, &lostEmpire.imageView);
	loadedTextures["empire_diffuse"] = lostEmpire;
	mainDeletionQueue.push_function([=]()
		{
			vkDestroyImageView(device, lostEmpire.imageView, nullptr);
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

void App::drawObjects(VkCommandBuffer commandBuffer, RenderObject* first, int count)
{
	glm::vec3 camPos = { 0.f, 0.f, -3.f };
	glm::mat4 view = glm::translate(glm::mat4(1.f), camPos);
	glm::mat4 projection = glm::perspective(glm::radians(70.f), 800.f / 600.f, 0.1f, 200.0f);
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
	float d = (frameNumber / 120.f);
	sceneParameters.ambientColor = { sin(d), 0 , cos(d), 1 };
	char* sceneData;
	vmaMapMemory(allocator, sceneParameterBuffer.allocation, (void**)&sceneData);
	int frameIndex = frameNumber % FRAME_OVERLAP;
	sceneData += padUniformBufferSize(sizeof(GPUSceneData)) * frameIndex;
	memcpy(sceneData, &sceneParameters, sizeof(GPUSceneData));
	vmaUnmapMemory(allocator, sceneParameterBuffer.allocation);

	// allocating object buffer
	void* objectData;
	vmaMapMemory(allocator, getCurrentFrame().objectBuffer.allocation, &objectData);
	GPUObjectData* objectSSBO = (GPUObjectData*)objectData;
	for (int i = 0; i < count; i++)
	{
		RenderObject& object = first[i];
		objectSSBO[i].modelMatrix = object.transformMatrix;
		// rotate
		objectSSBO[i].modelMatrix = glm::rotate(objectSSBO[i].modelMatrix, glm::radians(frameNumber * 0.5f), glm::vec3(0, 1, 0));
	}
	vmaUnmapMemory(allocator, getCurrentFrame().objectBuffer.allocation);

	Mesh* lastMesh = nullptr;
	Material* lastMaterial = nullptr;
	for (int i = 0; i < count; i++)
	{
		RenderObject& object = first[i];

		if (object.material != lastMaterial)
		{
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipeline);
			lastMaterial = object.material;
			// camera data descriptor
			uint32_t uniformOffset = padUniformBufferSize(sizeof(GPUSceneData)) * frameIndex;
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout,
				0, 1, &getCurrentFrame().globalDescriptor, 1, &uniformOffset);
			// object data descriptor
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout,
				1, 1, &getCurrentFrame().objectDescriptor, 0, nullptr);
		}

		glm::mat4 model = object.transformMatrix;
		model = glm::rotate(model, glm::radians(frameNumber * 0.5f), glm::vec3(0, 1, 0));

		MeshPushConstants constants;
		constants.renderMatrix = model;

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
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10}
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

	VkDescriptorSetLayoutBinding bindings[] = { camBinding, sceneBinding };

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

	const size_t sceneParameterBufferSize = FRAME_OVERLAP * padUniformBufferSize(sizeof(GPUSceneData));
	sceneParameterBuffer = createBuffer(sceneParameterBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

	for (int i = 0; i < FRAME_OVERLAP; i++)
	{
		frames[i].cameraBuffer = createBuffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		
		const int MAX_OBJECTS = 100000;
		frames[i].objectBuffer = createBuffer(sizeof(GPUObjectData) * MAX_OBJECTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

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

		// info about object buffer for descriptor
		VkDescriptorBufferInfo objectBufferInfo = {};
		objectBufferInfo.buffer = frames[i].objectBuffer.buffer;
		objectBufferInfo.offset = 0;
		objectBufferInfo.range = sizeof(GPUObjectData) * MAX_OBJECTS;

		// write descriptors
		VkWriteDescriptorSet cameraWrite = VkInit::writeDescriptorBuffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, frames[i].globalDescriptor, &cameraInfo, 0);
		VkWriteDescriptorSet sceneWrite = VkInit::writeDescriptorBuffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, frames[i].globalDescriptor, &sceneInfo, 1);
		VkWriteDescriptorSet objectWrite = VkInit::writeDescriptorBuffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, frames[i].objectDescriptor, &objectBufferInfo, 0);

		VkWriteDescriptorSet setWrites[] = { cameraWrite, sceneWrite, objectWrite };

		vkUpdateDescriptorSets(device, 3, setWrites, 0, nullptr);
	}

	// add buffers to deletion queues
	mainDeletionQueue.push_function([&]()
		{
			vmaDestroyBuffer(allocator, sceneParameterBuffer.buffer, sceneParameterBuffer.allocation);
			vkDestroyDescriptorSetLayout(device, objectSetLayout, nullptr);
			vkDestroyDescriptorSetLayout(device, globalSetLayout, nullptr);
			vkDestroyDescriptorPool(device, descriptorPool, nullptr);

			for (int i = 0; i < FRAME_OVERLAP; i++)
			{
				vmaDestroyBuffer(allocator, frames[i].cameraBuffer.buffer, frames[i].cameraBuffer.allocation);
				vmaDestroyBuffer(allocator, frames[i].objectBuffer.buffer, frames[i].objectBuffer.allocation);
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
