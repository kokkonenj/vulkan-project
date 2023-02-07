#include "app.h"
#include <SDL.h>
#include <SDL_vulkan.h>
#include <VkBootstrap.h>
#include <iostream>
#include <fstream>

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
	initPipelines();

	isInitialized = true;
}

App::~App()
{
	if (isInitialized)
	{
		vkDeviceWaitIdle(device); // wait for GPU to finish
		vkDestroyFence(device, renderFence, nullptr);
		vkDestroySemaphore(device, renderSemaphore, nullptr);
		vkDestroySemaphore(device, presentSemaphore, nullptr);
		vkDestroyCommandPool(device, commandPool, nullptr);
		vkDestroySwapchainKHR(device, swapchain, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);
		for (int i = 0; i < frameBuffers.size(); i++)
		{
			vkDestroyFramebuffer(device, frameBuffers[i], nullptr);
			vkDestroyImageView(device, swapchainImageViews[i], nullptr);
		}
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
	vkb::Device vkbDevice = deviceBuilder.build().value();
	device = vkbDevice.device;
	gpu = physicalDevice.physical_device;

	// Get graphics queue with vkBootstrap
	graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();
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
}

void App::initCommands()
{
	// Create command pool for commands submitted to the graphics queue
	VkCommandPoolCreateInfo commandPoolInfo = {};
	commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	commandPoolInfo.pNext = nullptr;
	commandPoolInfo.queueFamilyIndex = graphicsQueueFamily;
	commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	VK_CHECK(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool));

	// Allocate command buffer
	VkCommandBufferAllocateInfo cmdAllocInfo = {};
	cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cmdAllocInfo.pNext = nullptr;
	cmdAllocInfo.commandPool = commandPool;
	cmdAllocInfo.commandBufferCount = 1;
	cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &mainCommandBuffer));
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

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttatchmentReference;

	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &colorAttachment;
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;

	VK_CHECK(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
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
		frameBufferInfo.pAttachments = &swapchainImageViews[i];
		VK_CHECK(vkCreateFramebuffer(device, &frameBufferInfo, nullptr, &frameBuffers[i]));
	}
}

void App::initSyncStructures()
{
	// Create synchronization structures
	VkFenceCreateInfo fenceCreateInfo = {};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceCreateInfo.pNext = nullptr;
	fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
	VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &renderFence));

	// Create semaphores
	VkSemaphoreCreateInfo semaphoreCreateInfo = {};
	semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	semaphoreCreateInfo.pNext = nullptr;
	semaphoreCreateInfo.flags = 0;
	VK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &presentSemaphore));
	VK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &renderSemaphore));
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

	// Create pipeline layout that controls input and output of shaders
	VkPipelineLayoutCreateInfo pipelineLayoutInfo = VkInit::pipelineLayoutCreateInfo();
	VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &trianglePipelineLayout));

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

	// Build the pipeline
	pipelineBuilder.pipelineLayout = trianglePipelineLayout;
	trianglePipeline = pipelineBuilder.buildPipeline(device, renderPass);
}

void App::draw()
{
	if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED)
	{
		return;
	}
	// Wait until GPU has finished rendering last frame
	VK_CHECK(vkWaitForFences(device, 1, &renderFence, true, 1000000000));
	VK_CHECK(vkResetFences(device, 1, &renderFence));

	// Request image from the swapchain
	uint32_t swapchainImageIndex;
	VK_CHECK(vkAcquireNextImageKHR(device, swapchain, 1000000000, presentSemaphore, nullptr, &swapchainImageIndex));

	// Reset command buffer
	VK_CHECK(vkResetCommandBuffer(mainCommandBuffer, 0));

	// Begin command buffer recording
	VkCommandBufferBeginInfo cmdBeginInfo = {};
	cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmdBeginInfo.pNext = nullptr;
	cmdBeginInfo.pInheritanceInfo = nullptr;
	cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VK_CHECK(vkBeginCommandBuffer(mainCommandBuffer, &cmdBeginInfo));

	// Color of the screen
	VkClearValue clearValue;
	float red = abs(sin(frameNumber / 120.f));
	float blue = abs(sin(frameNumber / 120.f + 1.57f));
	clearValue.color = { red, 0.0f, blue, 1.0f };
	
	// Starting the renderpass
	VkRenderPassBeginInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassInfo.pNext = nullptr;
	renderPassInfo.renderPass = renderPass;
	renderPassInfo.renderArea.offset.x = 0;
	renderPassInfo.renderArea.offset.y = 0;
	renderPassInfo.renderArea.extent = windowExtent;
	renderPassInfo.framebuffer = frameBuffers[swapchainImageIndex];
	renderPassInfo.clearValueCount = 1;
	renderPassInfo.pClearValues = &clearValue;
	vkCmdBeginRenderPass(mainCommandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	/* ----- RENDERING COMMANDS BEGIN ----- */
	vkCmdBindPipeline(mainCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, trianglePipeline);
	vkCmdDraw(mainCommandBuffer, 3, 1, 0, 0);
	/* ----- RENDERING COMMANDS END ----- */

	// Finalize render pass and command buffer
	vkCmdEndRenderPass(mainCommandBuffer);
	VK_CHECK(vkEndCommandBuffer(mainCommandBuffer));

	// submit image to the queue
	VkSubmitInfo submit = {};
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.pNext = nullptr;
	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	submit.pWaitDstStageMask = &waitStage;
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &presentSemaphore;
	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &renderSemaphore;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &mainCommandBuffer;
	VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submit, renderFence));

	// put rendered image to visible window
	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext = nullptr;
	presentInfo.pSwapchains = &swapchain;
	presentInfo.swapchainCount = 1;
	presentInfo.pWaitSemaphores = &renderSemaphore;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pImageIndices = &swapchainImageIndex;
	VK_CHECK(vkQueuePresentKHR(graphicsQueue, &presentInfo));

	frameNumber++;
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
