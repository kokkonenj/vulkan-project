#include <vk_textures.h>
#include <iostream>
#include "vk_initializers.h"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

bool utils::loadImageFromFile(App* app, const char* file, AllocatedImage& outImage, VkFormat format)
{
	int texWidth, texHeight, texChannels;
	stbi_uc* pixels = stbi_load(file, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	if (!pixels)
	{
		std::cout << "Failed to load texture file :" << file << std::endl;
		return false;
	}

	void* pPixels = pixels;
	VkDeviceSize imageSize = texWidth * texHeight * 4;
	VkFormat imageFormat = format;
	AllocatedBuffer stagingBuffer = app->createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

	void* data;
	vmaMapMemory(app->allocator, stagingBuffer.allocation, &data);
	memcpy(data, pPixels, static_cast<size_t>(imageSize));
	vmaUnmapMemory(app->allocator, stagingBuffer.allocation);
	stbi_image_free(pixels);

	VkExtent3D imageExtent;
	imageExtent.width = static_cast<uint32_t>(texWidth);
	imageExtent.height = static_cast<uint32_t>(texHeight);
	imageExtent.depth = 1;
	VkImageCreateInfo dImgInfo = VkInit::imageCreateInfo(imageFormat, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, imageExtent, VK_SAMPLE_COUNT_1_BIT);
	AllocatedImage newImage;
	VmaAllocationCreateInfo dImgAllocInfo = {};
	dImgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	vmaCreateImage(app->allocator, &dImgInfo, &dImgAllocInfo, &newImage.image, &newImage.allocation, nullptr);

	app->immediateSubmit([&](VkCommandBuffer cmd) {
		VkImageSubresourceRange range = {};
		range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		range.baseMipLevel = 0;
		range.levelCount = 1;
		range.baseArrayLayer = 0;
		range.layerCount = 1;

		VkImageMemoryBarrier imageBarriertoTransfer = {};
		imageBarriertoTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageBarriertoTransfer.pNext = nullptr;
		imageBarriertoTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageBarriertoTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageBarriertoTransfer.image = newImage.image;
		imageBarriertoTransfer.subresourceRange = range;
		imageBarriertoTransfer.srcAccessMask = 0;
		imageBarriertoTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 
			0, 0, nullptr, 0, nullptr, 1, &imageBarriertoTransfer);

		VkBufferImageCopy copyRegion = {};
		copyRegion.bufferOffset = 0;
		copyRegion.bufferRowLength = 0;
		copyRegion.bufferImageHeight = 0;
		copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyRegion.imageSubresource.mipLevel = 0;
		copyRegion.imageSubresource.baseArrayLayer = 0;
		copyRegion.imageSubresource.layerCount = 1;
		copyRegion.imageExtent = imageExtent;
		vkCmdCopyBufferToImage(cmd, stagingBuffer.buffer, newImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

		VkImageMemoryBarrier imageBarriertoReadable = imageBarriertoTransfer;
		imageBarriertoReadable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageBarriertoReadable.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageBarriertoReadable.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		imageBarriertoReadable.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
			nullptr, 0, nullptr, 1, &imageBarriertoReadable);

		});

	app->mainDeletionQueue.push_function([=]()
		{
			vmaDestroyImage(app->allocator, newImage.image, newImage.allocation);
			vmaDestroyBuffer(app->allocator, stagingBuffer.buffer, stagingBuffer.allocation);
		});
	std::cout << "Texture loaded successfully: " << file << std::endl;
	outImage = newImage;
	return true;
}

bool utils::loadImageFromBuffer(App* app, void* buffer, AllocatedImage& outImage, VkFormat format, uint32_t texWidth, uint32_t texHeight)
{
	if (!buffer)
	{
		std::cout << "Failed to load texture from buffer" << std::endl;
		return false;
	}

	void* pPixels = buffer;
	VkDeviceSize imageSize = texWidth * texHeight * 4;
	VkFormat imageFormat = format;
	AllocatedBuffer stagingBuffer = app->createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

	void* data;
	vmaMapMemory(app->allocator, stagingBuffer.allocation, &data);
	memcpy(data, pPixels, static_cast<size_t>(imageSize));
	vmaUnmapMemory(app->allocator, stagingBuffer.allocation);

	VkExtent3D imageExtent;
	imageExtent.width = static_cast<uint32_t>(texWidth);
	imageExtent.height = static_cast<uint32_t>(texHeight);
	imageExtent.depth = 1;
	VkImageCreateInfo dImgInfo = VkInit::imageCreateInfo(imageFormat, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, imageExtent, VK_SAMPLE_COUNT_1_BIT);
	AllocatedImage newImage;
	VmaAllocationCreateInfo dImgAllocInfo = {};
	dImgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	vmaCreateImage(app->allocator, &dImgInfo, &dImgAllocInfo, &newImage.image, &newImage.allocation, nullptr);

	app->immediateSubmit([&](VkCommandBuffer cmd) {
		VkImageSubresourceRange range = {};
		range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		range.baseMipLevel = 0;
		range.levelCount = 1;
		range.baseArrayLayer = 0;
		range.layerCount = 1;

		VkImageMemoryBarrier imageBarriertoTransfer = {};
		imageBarriertoTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageBarriertoTransfer.pNext = nullptr;
		imageBarriertoTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageBarriertoTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageBarriertoTransfer.image = newImage.image;
		imageBarriertoTransfer.subresourceRange = range;
		imageBarriertoTransfer.srcAccessMask = 0;
		imageBarriertoTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &imageBarriertoTransfer);

		VkBufferImageCopy copyRegion = {};
		copyRegion.bufferOffset = 0;
		copyRegion.bufferRowLength = 0;
		copyRegion.bufferImageHeight = 0;
		copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyRegion.imageSubresource.mipLevel = 0;
		copyRegion.imageSubresource.baseArrayLayer = 0;
		copyRegion.imageSubresource.layerCount = 1;
		copyRegion.imageExtent = imageExtent;
		vkCmdCopyBufferToImage(cmd, stagingBuffer.buffer, newImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

		VkImageMemoryBarrier imageBarriertoReadable = imageBarriertoTransfer;
		imageBarriertoReadable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageBarriertoReadable.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageBarriertoReadable.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		imageBarriertoReadable.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
			nullptr, 0, nullptr, 1, &imageBarriertoReadable);

		});

	app->mainDeletionQueue.push_function([=]()
		{
			vmaDestroyImage(app->allocator, newImage.image, newImage.allocation);
			vmaDestroyBuffer(app->allocator, stagingBuffer.buffer, stagingBuffer.allocation);
		});
	std::cout << "Texture loaded successfully from buffer" << std::endl;
	outImage = newImage;
	return true;
}
