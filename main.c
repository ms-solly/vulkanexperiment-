#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#define STB_DS_IMPLEMENTATION
#include "external/stb/stb_ds.h"
#define VK_NO_PROTOTYPES
#define VOLK_IMPLEMENTATION
#define GLFW_INCLUDE_VULKAN
#define STB_IMAGE_IMPLEMENTATION

#include "external/stb/stb_image.h"

#include <GLFW/glfw3.h>
#define CGLTF_IMPLEMENTATION
#include "external/cgltf/cgltf.h"

#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_WAYLAND
#include <GLFW/glfw3native.h>
#define FAST_OBJ_IMPLEMENTATION
#include "external/fast_obj/fast_obj.h"
#include "external/volk/volk.h"

#include "external/cglm/include/cglm/cglm.h"
// #include "external/tracy/public/tracy/TracyC.h"

// Nuklear GUI includes

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_IMPLEMENTATION
#define NK_GLFW_VULKAN_IMPLEMENTATION

#include "external/nuklear/nuklear.h"

#include "external/nuklear/demo/glfw_vulkan/nuklear_glfw_vulkan.h"

#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_WAYLAND
#include <GLFW/glfw3native.h>

#include "tinytypes.h"

#define VK_CHECK(call) \
	do \
	{ \
		VkResult result_ = call; \
		if (result_ != VK_SUCCESS) \
		{ \
			fprintf(stderr, "Vulkan call failed in %s:%d with error %d\n", __FILE__, __LINE__, result_); \
			assert(result_ == VK_SUCCESS); \
		} \
	} while (0)
#ifndef ARRAYSIZE
#define ARRAYSIZE(array) (sizeof(array) / sizeof((array)[0]))
#endif

// --- Core Structures ---

typedef struct
{
	vec3 pos;
	vec3 normal;
	vec2 texcoord;
	vec4 color; // 👈 Per-vertex color from material
} Vertex;

typedef struct Buffer
{
	VkBuffer vkbuffer;
	VkDeviceMemory memory;
	void* data;
	size_t size;
} Buffer;

typedef struct Texture
{
	VkImage image;
	VkDeviceMemory memory;
	VkImageView view;
	VkSampler sampler;
} Texture;

typedef struct Material
{
	vec4 base_color; // Base color factor
	int has_texture;
	int texture_index; // Index into the textures array (-1 if no texture)
	char* texture_path;
} Material;

typedef struct Primitive
{
	u32 first_index;    // Starting index in the index buffer
	u32 index_count;    // Number of indices for this primitive
	int material_index; // Index into the materials array
} Primitive;

#define MAX_POINT_LIGHTS 8

typedef struct PointLight
{
	vec3 position;
	float _pad1; // Padding for alignment
	vec3 color;
	float intensity;
	float constant; // Attenuation factors
	float linear;
	float quadratic;
	float _pad2; // Padding for alignment
} PointLight;

typedef struct UniformBufferObject
{
	mat4 proj;
	mat4 view;
	mat4 model;
	vec3 cameraPos;
	u32 numLights;
	PointLight lights[MAX_POINT_LIGHTS];
} UniformBufferObject;

typedef struct Mesh
{
	Vertex* vertices;
	u32* indices;
	u32 vertex_count;
	u32 index_count;

	// Per-material texture support
	Material* materials;
	u32 material_count;
	Primitive* primitives;
	u32 primitive_count;

	// Legacy fields (kept for compatibility)
	char* texture_path; // from glTF material texture
	vec4 base_color;    // from glTF material baseColorFactor
	int has_texture;    // 1 if texture used, else 0
} Mesh;

typedef struct GrassInstance
{
	vec3 position; // World position
	float scale;   // Size variation
} GrassInstance;

typedef struct GrassField
{
	GrassInstance* instances;
	u32 instance_count;
	Buffer instanceBuffer;

	// Grass rendering pipeline
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
	VkShaderModule vertShaderModule;
	VkShaderModule fragShaderModule;
} GrassField;

#define MAX_FRAMES_IN_FLIGHT 2

typedef struct Application
{
	GLFWwindow* window;
	int width, height;
	bool framebufferResized;
	VkDebugUtilsMessengerEXT DebugMessenger;

	// Camera
	vec3 cameraPos;
	vec3 cameraFront;
	vec3 cameraUp;
	float yaw;
	float pitch;
	float lastX;
	float lastY;
	bool firstMouse;
	float deltaTime;
	float lastFrame;

	// Vulkan core
	VkInstance instance;
	VkPhysicalDevice physicalDevice;
	VkDevice device;
	VkQueue graphicsQueue;
	VkCommandPool commandPool;
	VkCommandBuffer* commandBuffers;
	VkPhysicalDeviceMemoryProperties memoryProperties;
	VkSurfaceKHR surface;

	// Swapchain
	VkSwapchainKHR swapchain;
	VkFormat swapchainFormat;
	VkColorSpaceKHR swapchainColorSpace;
	VkImage* swapchainImages;
	VkImageView* swapchainImageViews;
	u32 swapchainImageCount;
	VkFramebuffer* framebuffers;

	// Depth buffer
	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;
	VkFormat depthFormat;

	// Command pool and buffers
	Buffer baseColorBuffer;
	Buffer hasTextureBuffer;

	// Pipeline
	VkRenderPass renderPass;
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	VkShaderModule vertShaderModule;
	VkShaderModule fragShaderModule;

	// Resources
	Mesh mesh;
	Buffer vertexBuffer;
	Buffer indexBuffer;

	// Multiple textures support
	Texture* textures;               // Array of textures
	u32 texture_count;               // Number of textures
	VkDescriptorSet* descriptorSets; // One descriptor set per texture

	// Legacy single texture support (kept for compatibility)
	Texture texture;
	u32 mipLevels;
	Buffer uniformBuffer;

	// Infinite grid
	VkPipeline gridPipeline;
	VkPipelineLayout gridPipelineLayout;
	VkShaderModule gridVertShaderModule;
	VkShaderModule gridFragShaderModule;
	Buffer gridVertexBuffer;
	Texture gridTexture;
	VkDescriptorSetLayout gridDescriptorSetLayout;
	VkDescriptorSet gridDescriptorSet;

	// Grass system
	GrassField grassField;

	// Descriptors
	VkDescriptorPool descriptorPool;
	VkDescriptorSet descriptorSet;
	VkPhysicalDeviceMemoryProperties memProperties;
	// VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;

	// Lighting
	PointLight lights[MAX_POINT_LIGHTS];
	u32 numActiveLights;

	// Sync objects

	VkSemaphore imageAvailableSemaphores[MAX_FRAMES_IN_FLIGHT]; // Per frame in flight
	VkSemaphore renderFinishedSemaphores[MAX_FRAMES_IN_FLIGHT]; // Per frame in flight
	VkFence inFlightFences[MAX_FRAMES_IN_FLIGHT];               // Per frame in flight
	u32 currentFrame;

} Application;

// --- Memory/Buffer Helpers ---

u32 selectmemorytype(
    VkPhysicalDeviceMemoryProperties* memprops, u32 memtypeBits, VkFlags requirements_mask);
void createBuffer(Application* app, Buffer* buffer, size_t size, VkBufferUsageFlags usage);
void destroyBuffer(VkDevice device, Buffer* buffer);

// --- Command Buffer Helpers ---
VkCommandBuffer beginSingleTimeCommands(Application* app);
void endSingleTimeCommands(Application* app, VkCommandBuffer commandBuffer);

// --- Texture Helpers ---

void generateMipmaps(Application* app, VkCommandBuffer commandBuffer, VkImage image, VkFormat imageFormat, i32 texWidth, i32 texHeight, u32 mipLevels);
void createTextureImage(Application* app, const char* path, Texture* outTexture, u32* outMipLevels);
void createTextureSampler(Application* app, Texture* texture, u32 mipLevels);

// --- Model Loading ---
void loadGltfModel(const char* path, Mesh* outMesh);
void createGroundPlane(Mesh* outMesh);

// Grass system functions
void createGrassField(Application* app);
void renderGrass(Application* app, VkCommandBuffer commandBuffer);
void cleanupGrass(Application* app);
uint32_t findMemoryType(const VkPhysicalDeviceMemoryProperties* memProperties, uint32_t typeFilter, VkMemoryPropertyFlags properties);
void checkMaterials(cgltf_data* data);

// --- Vulkan Helpers ---
VkInstance createVulkanInstance(void);
VkPhysicalDevice selectPhysicalDevice(VkInstance instance);
u32 find_graphics_queue_family_index(VkPhysicalDevice pickedPhysicalDevice);
VkDevice create_logical_device(VkPhysicalDevice pickedPhysicaldevice, u32 queueFamilyIndex);
VkSurfaceKHR createSurface(VkInstance instance, GLFWwindow* window);
VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device);
VkDescriptorSetLayout createGridDescriptorSetLayout(VkDevice device);
VkDescriptorPool createDescriptorPool(VkDevice device);
VkDescriptorSet allocateDescriptorSet(VkDevice device, VkDescriptorPool pool, VkDescriptorSetLayout layout);
VkSwapchainKHR createSwapchain(Application* app);
VkRenderPass createRenderPass(VkDevice device, VkFormat colorFormat, VkFormat inDepthFormat);
void createSwapchainViews(Application* app);
void createFramebuffers(Application* app);
VkShaderModule LoadShaderModule(const char* filepath, VkDevice device);
VkFormat findDepthFormat(VkPhysicalDevice physicalDevice);
void createDepthResources(Application* app);
VkPipeline createGraphicsPipeline(Application* app, VkShaderModule vertShader, VkShaderModule fragShader);
VkPipeline createGridPipeline(Application* app, VkShaderModule vertShader, VkShaderModule fragShader);
VkPipeline createGrassPipeline(Application* app, VkShaderModule vertShader, VkShaderModule fragShader);
void createSwapchainRelatedResources(Application* app);
void createModelAndBuffers(Application* app);
void createTextureResources(Application* app);
void createUniformBuffer(Application* app);
void createDescriptors(Application* app);
void createSyncObjects(Application* app);
void recordCommandBuffer(Application* app, VkCommandBuffer commandBuffer, u32 imageIndex);
void updateLights(Application* app);

void drawFrame(Application* app);
void cleanupSwapchain(Application* app);
void recreateSwapchain(Application* app);

// --- Vulkan Initialization Helpers ---
void createCommandPoolAndBuffer(Application* app, u32 queueFamilyIndex);
void createPipeline(Application* app);
void createResources(Application* app);

// --- Vulkan Cleanup Helpers ---
void cleanupSyncObjects(Application* app);
void cleanupResources(Application* app);
void cleanupPipeline(Application* app);

// --- Main Application ---
void framebufferResizeCallback(GLFWwindow* window, int width, int height);
void initWindow(Application* app);
void initVulkan(Application* app);
void mainLoop(Application* app);
void cleanup(Application* app);
int main(void);

// --- Input Handling ---
void processInput(Application* app);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);

// --- Command Buffer Helpers ---

VkCommandBuffer beginSingleTimeCommands(Application* app)
{
	VkCommandBufferAllocateInfo allocInfo = {
	    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
	    .commandPool = app->commandPool,
	    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
	    .commandBufferCount = 1,
	};
	VkCommandBuffer commandBuffer;
	VK_CHECK(vkAllocateCommandBuffers(app->device, &allocInfo, &commandBuffer));

	VkCommandBufferBeginInfo beginInfo = {
	    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
	return commandBuffer;
}

void endSingleTimeCommands(Application* app, VkCommandBuffer commandBuffer)
{
	VK_CHECK(vkEndCommandBuffer(commandBuffer));

	VkSubmitInfo submitInfo = {
	    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
	    .commandBufferCount = 1,
	    .pCommandBuffers = &commandBuffer,
	};
	VK_CHECK(vkQueueSubmit(app->graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE));
	VK_CHECK(vkQueueWaitIdle(app->graphicsQueue));

	vkFreeCommandBuffers(app->device, app->commandPool, 1, &commandBuffer);
}

void generateMipmaps(Application* app, VkCommandBuffer cmd, VkImage image, VkFormat format, int32_t texWidth, int32_t texHeight, uint32_t mipLevels)
{
	int32_t mipWidth = texWidth;
	int32_t mipHeight = texHeight;

	for (uint32_t i = 1; i < mipLevels; i++)
	{
		VkImageMemoryBarrier barrierBeforeBlit = {
		    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		    .image = image,
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		    .subresourceRange.baseArrayLayer = 0,
		    .subresourceRange.layerCount = 1,
		    .subresourceRange.levelCount = 1,
		    .subresourceRange.baseMipLevel = i - 1,
		    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
		};

		vkCmdPipelineBarrier(
		    cmd,
		    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		    0,
		    0, NULL,
		    0, NULL,
		    1, &barrierBeforeBlit);

		VkImageBlit blit = {
		    .srcOffsets[1] = {mipWidth, mipHeight, 1},
		    .srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		    .srcSubresource.mipLevel = i - 1,
		    .srcSubresource.baseArrayLayer = 0,
		    .srcSubresource.layerCount = 1,

		    .dstOffsets[1] = {
		        mipWidth > 1 ? mipWidth / 2 : 1,
		        mipHeight > 1 ? mipHeight / 2 : 1,
		        1},
		    .dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		    .dstSubresource.mipLevel = i,
		    .dstSubresource.baseArrayLayer = 0,
		    .dstSubresource.layerCount = 1,
		};

		vkCmdBlitImage(
		    cmd,
		    image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		    image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		    1, &blit,
		    VK_FILTER_LINEAR);

		// Transition previous mip level to SHADER_READ_ONLY_OPTIMAL
		VkImageMemoryBarrier barrierAfterBlit = barrierBeforeBlit;
		barrierAfterBlit.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrierAfterBlit.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrierAfterBlit.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barrierAfterBlit.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(
		    cmd,
		    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		    0,
		    0, NULL,
		    0, NULL,
		    1, &barrierAfterBlit);

		if (mipWidth > 1)
			mipWidth /= 2;
		if (mipHeight > 1)
			mipHeight /= 2;
	}

	// Transition last mip level to SHADER_READ_ONLY_OPTIMAL
	VkImageMemoryBarrier lastBarrier = {
	    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
	    .image = image,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
	    .subresourceRange.baseMipLevel = mipLevels - 1,
	    .subresourceRange.levelCount = 1,
	    .subresourceRange.baseArrayLayer = 0,
	    .subresourceRange.layerCount = 1,
	    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	    .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
	    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
	};

	vkCmdPipelineBarrier(
	    cmd,
	    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
	    0,
	    0, NULL,
	    0, NULL,
	    1, &lastBarrier);
}

void createTextureImage(Application* app, const char* path, Texture* outTexture, u32* outMipLevels)
{

	int texWidth, texHeight, texChannels;
	stbi_uc* pixels = stbi_load(path, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	VkDeviceSize imageSize = texWidth * texHeight * 4;

	*outMipLevels = (u32)(floor(log2(texWidth > texHeight ? texWidth : texHeight))) + 1;

	if (!pixels)
	{
		fprintf(stderr, "Failed to load texture image: %s\n", path);
		fprintf(stderr, "STB Error: %s\n", stbi_failure_reason());
		exit(1);
	}

	printf("Loaded texture: %s (%dx%d, %d channels, %u mip levels)\n",
	    path, texWidth, texHeight, texChannels, *outMipLevels);

	Buffer stagingBuffer;
	createBuffer(app, &stagingBuffer, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	memcpy(stagingBuffer.data, pixels, (size_t)imageSize);
	stbi_image_free(pixels);

	VkImageCreateInfo imageInfo = {
	    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
	    .imageType = VK_IMAGE_TYPE_2D,
	    .extent.width = texWidth,
	    .extent.height = texHeight,
	    .extent.depth = 1,
	    .mipLevels = *outMipLevels,
	    .arrayLayers = 1,
	    .format = VK_FORMAT_R8G8B8A8_SRGB,
	    .tiling = VK_IMAGE_TILING_OPTIMAL,
	    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	    .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
	    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	    .samples = VK_SAMPLE_COUNT_1_BIT,
	};

	VK_CHECK(vkCreateImage(app->device, &imageInfo, NULL, &outTexture->image));

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(app->device, outTexture->image, &memRequirements);

	VkMemoryAllocateInfo allocInfo = {
	    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	    .allocationSize = memRequirements.size,
	    .memoryTypeIndex = selectmemorytype(&app->memoryProperties, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
	};

	VK_CHECK(vkAllocateMemory(app->device, &allocInfo, NULL, &outTexture->memory));
	VK_CHECK(vkBindImageMemory(app->device, outTexture->image, outTexture->memory, 0));

	VkCommandBuffer commandBuffer = beginSingleTimeCommands(app);

	// Transition layout to TRANSFER_DST_OPTIMAL
	VkImageMemoryBarrier barrier = {
	    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
	    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	    .image = outTexture->image,
	    .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
	    .subresourceRange.baseMipLevel = 0,
	    .subresourceRange.levelCount = *outMipLevels,
	    .subresourceRange.baseArrayLayer = 0,
	    .subresourceRange.layerCount = 1,
	    .srcAccessMask = 0,
	    .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
	};

	vkCmdPipelineBarrier(
	    commandBuffer,
	    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
	    0,
	    0, NULL,
	    0, NULL,
	    1, &barrier);

	// Copy buffer to image
	VkBufferImageCopy region = {
	    .bufferOffset = 0,
	    .bufferRowLength = 0,
	    .bufferImageHeight = 0,
	    .imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
	    .imageSubresource.mipLevel = 0,
	    .imageSubresource.baseArrayLayer = 0,
	    .imageSubresource.layerCount = 1,
	    .imageOffset = {0, 0, 0},
	    .imageExtent = {(u32)texWidth, (u32)texHeight, 1},
	};
	vkCmdCopyBufferToImage(commandBuffer, stagingBuffer.vkbuffer, outTexture->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	// Generate mipmaps and transition layout to shader read
	generateMipmaps(app, commandBuffer, outTexture->image, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, *outMipLevels);

	endSingleTimeCommands(app, commandBuffer);

	destroyBuffer(app->device, &stagingBuffer);

	// Create image view
	VkImageViewCreateInfo viewInfo = {
	    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
	    .image = outTexture->image,
	    .viewType = VK_IMAGE_VIEW_TYPE_2D,
	    .format = VK_FORMAT_R8G8B8A8_SRGB,
	    .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
	    .subresourceRange.baseMipLevel = 0,
	    .subresourceRange.levelCount = *outMipLevels,
	    .subresourceRange.baseArrayLayer = 0,
	    .subresourceRange.layerCount = 1,
	};
	VK_CHECK(vkCreateImageView(app->device, &viewInfo, NULL, &outTexture->view));
}
void updateBaseColorAndHasTexture(Application* app)
{
	// ✅ Use already mapped pointer for baseColor
	memcpy(app->baseColorBuffer.data, app->mesh.base_color, sizeof(vec4));

	// ✅ Use already mapped pointer for hasTexture
	int hasTexture = app->mesh.has_texture;
	memcpy(app->hasTextureBuffer.data, &hasTexture, sizeof(int));
}

void createTextureSampler(Application* app, Texture* texture, u32 mipLevels)
{
	VkPhysicalDeviceProperties properties = {0};
	vkGetPhysicalDeviceProperties(app->physicalDevice, &properties);

	VkSamplerCreateInfo samplerInfo = {
	    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
	    .magFilter = VK_FILTER_LINEAR,
	    .minFilter = VK_FILTER_LINEAR,
	    .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
	    .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
	    .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
	    .anisotropyEnable = VK_TRUE,
	    .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
	    .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
	    .unnormalizedCoordinates = VK_FALSE,
	    .compareEnable = VK_FALSE,
	    .compareOp = VK_COMPARE_OP_ALWAYS,
	    .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
	    .mipLodBias = 0.0f,
	    .minLod = 0.0f,
	    .maxLod = (float)mipLevels,
	};

	VK_CHECK(vkCreateSampler(app->device, &samplerInfo, NULL, &texture->sampler));
}

uint32_t findMemoryType(const VkPhysicalDeviceMemoryProperties* memProperties, uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
	for (uint32_t i = 0; i < memProperties->memoryTypeCount; i++)
	{
		if ((typeFilter & (1 << i)) &&
		    (memProperties->memoryTypes[i].propertyFlags & properties) == properties)
		{
			return i;
		}
	}

	fprintf(stderr, "Failed to find suitable memory type!\n");
	exit(1);
}

void processNode(cgltf_node* node, Mesh* outMesh, cgltf_data* data, mat4 parentTransform, Application* app, uint32_t* vertexOffset, uint32_t* indexOffset, uint32_t* primitiveIndex)
{
	mat4 localTransform;
	glm_mat4_identity(localTransform);
	cgltf_node_transform_local(node, (cgltf_float*)localTransform);

	mat4 worldTransform;
	glm_mat4_mul(parentTransform, localTransform, worldTransform);

	if (node->mesh)
	{
		for (cgltf_size i = 0; i < node->mesh->primitives_count; ++i)
		{
			cgltf_primitive* primitive = &node->mesh->primitives[i];
			uint32_t startVertex = *vertexOffset;
			uint32_t startIndex = *indexOffset;
			cgltf_accessor* posAccessor = NULL;

			// Find material index
			int materialIndex = -1;
			if (primitive->material)
			{
				materialIndex = (int)(primitive->material - data->materials);
			}

			// Get base color from material
			vec4 baseColor = {1.0f, 1.0f, 1.0f, 1.0f}; // default white
			if (materialIndex >= 0 && materialIndex < (int)outMesh->material_count)
			{
				memcpy(baseColor, outMesh->materials[materialIndex].base_color, sizeof(vec4));
			}

			// Find POSITION accessor
			for (cgltf_size a = 0; a < primitive->attributes_count; ++a)
			{
				if (strcmp(primitive->attributes[a].name, "POSITION") == 0)
				{
					posAccessor = primitive->attributes[a].data;
					break;
				}
			}

			if (!posAccessor)
				continue;

			// Extract vertices
			for (cgltf_size v = 0; v < posAccessor->count; ++v)
			{
				Vertex vert = {0};

				for (cgltf_size a = 0; a < primitive->attributes_count; ++a)
				{
					cgltf_attribute* attr = &primitive->attributes[a];
					float* buffer = (float*)attr->data->buffer_view->buffer->data +
					                attr->data->buffer_view->offset / sizeof(float) +
					                attr->data->offset / sizeof(float);

					if (strcmp(attr->name, "POSITION") == 0)
					{
						vec4 pos = {buffer[v * 3], buffer[v * 3 + 1], buffer[v * 3 + 2], 1.0f};
						vec4 transformed;
						glm_mat4_mulv(worldTransform, pos, transformed);
						glm_vec3_copy(transformed, vert.pos);
					}
					else if (strcmp(attr->name, "NORMAL") == 0)
					{
						memcpy(&vert.normal, buffer + v * 3, sizeof(vec3));
					}
					else if (strcmp(attr->name, "TEXCOORD_0") == 0)
					{
						memcpy(&vert.texcoord, buffer + v * 2, sizeof(vec2));
					}
				}

				// Set vertex color from material
				memcpy(vert.color, baseColor, sizeof(vec4));

				outMesh->vertices[(*vertexOffset)++] = vert;
			}

			// Add indices and create primitive entry
			uint32_t indexCount = 0;
			if (primitive->indices)
			{
				indexCount = primitive->indices->count;
				for (cgltf_size k = 0; k < primitive->indices->count; ++k)
				{
					outMesh->indices[(*indexOffset)++] = startVertex + cgltf_accessor_read_index(primitive->indices, k);
				}
			}
			else
			{
				indexCount = posAccessor->count;
				for (cgltf_size k = 0; k < posAccessor->count; ++k)
				{
					outMesh->indices[(*indexOffset)++] = startVertex + k;
				}
			}

			// Create primitive entry
			if (*primitiveIndex < outMesh->primitive_count)
			{
				outMesh->primitives[*primitiveIndex].first_index = startIndex;
				outMesh->primitives[*primitiveIndex].index_count = indexCount;
				outMesh->primitives[*primitiveIndex].material_index = materialIndex;
				(*primitiveIndex)++;
			}
		}
	}

	// Recurse through children
	for (cgltf_size i = 0; i < node->children_count; ++i)
	{
		processNode(node->children[i], outMesh, data, worldTransform, app, vertexOffset, indexOffset, primitiveIndex);
	}
}

void checkMaterials(cgltf_data* data)
{
	printf("Total materials defined in file: %zu\n", data->materials_count);

	size_t usedMaterialCount = 0;
	bool usedMaterialFlags[256] = {false}; // assuming max 256 materials

	for (size_t m = 0; m < data->meshes_count; ++m)
	{
		const cgltf_mesh* mesh = &data->meshes[m];

		for (size_t p = 0; p < mesh->primitives_count; ++p)
		{
			const cgltf_primitive* prim = &mesh->primitives[p];
			if (prim->material)
			{
				size_t materialIndex = prim->material - data->materials;

				if (!usedMaterialFlags[materialIndex])
				{
					usedMaterialFlags[materialIndex] = true;
					usedMaterialCount++;

					printf("Used material %zu: %s\n",
					    materialIndex,
					    prim->material->name ? prim->material->name : "(unnamed)");

					// Print base color factor
					const float* color = prim->material->pbr_metallic_roughness.base_color_factor;
					printf("  Base Color: (%.2f, %.2f, %.2f, %.2f)\n", color[0], color[1], color[2], color[3]);
				}
			}
			else
			{
				printf("Primitive has no material assigned.\n");
			}
		}
	}

	printf("Total materials actually used in primitives: %zu\n", usedMaterialCount);
}

// --- Model Loading ---

void loadGltfModel(const char* path, Mesh* outMesh)
{
	cgltf_options options = {0};
	cgltf_data* data = NULL;

	cgltf_parse_file(&options, path, &data);

	char* dir_path = NULL;
	char* last_slash = strrchr(path, '/');
	if (last_slash)
	{
		size_t dir_len = last_slash - path + 1;
		dir_path = malloc(dir_len + 1);
		strncpy(dir_path, path, dir_len);
		dir_path[dir_len] = '\0';
	}
	else
	{
		dir_path = malloc(3);
		strcpy(dir_path, "./");
	}

	cgltf_load_buffers(&options, data, dir_path);

	// Initialize mesh
	memset(outMesh, 0, sizeof(Mesh));

	// Parse materials first
	outMesh->material_count = data->materials_count;
	if (outMesh->material_count > 0)
	{
		outMesh->materials = calloc(outMesh->material_count, sizeof(Material));

		for (cgltf_size i = 0; i < data->materials_count; ++i)
		{
			cgltf_material* mat = &data->materials[i];
			Material* ourMat = &outMesh->materials[i];

			// Initialize with default values
			ourMat->base_color[0] = 1.0f;
			ourMat->base_color[1] = 1.0f;
			ourMat->base_color[2] = 1.0f;
			ourMat->base_color[3] = 1.0f;
			ourMat->has_texture = 0;
			ourMat->texture_index = -1;
			ourMat->texture_path = NULL;

			if (mat->has_pbr_metallic_roughness)
			{
				cgltf_pbr_metallic_roughness* pbr = &mat->pbr_metallic_roughness;
				memcpy(ourMat->base_color, pbr->base_color_factor, sizeof(float) * 4);

				if (pbr->base_color_texture.texture &&
				    pbr->base_color_texture.texture->image &&
				    pbr->base_color_texture.texture->image->uri)
				{

					const char* uri = pbr->base_color_texture.texture->image->uri;
					size_t full_path_len = strlen(dir_path) + strlen(uri) + 1;
					ourMat->texture_path = malloc(full_path_len);
					snprintf(ourMat->texture_path, full_path_len, "%s%s", dir_path, uri);
					ourMat->has_texture = 1;
				}
			}
		}
	}

	// Count vertices, indices, and primitives
	size_t totalVertices = 0, totalIndices = 0, totalPrimitives = 0;
	for (cgltf_size i = 0; i < data->meshes_count; ++i)
	{
		cgltf_mesh* mesh = &data->meshes[i];
		for (cgltf_size j = 0; j < mesh->primitives_count; ++j)
		{
			cgltf_primitive* prim = &mesh->primitives[j];
			if (prim->attributes_count == 0)
				continue;

			totalVertices += prim->attributes[0].data->count;
			totalIndices += prim->indices ? prim->indices->count : prim->attributes[0].data->count;
			totalPrimitives++;
		}
	}

	// Allocate arrays
	outMesh->vertex_count = totalVertices;
	outMesh->index_count = totalIndices;
	outMesh->primitive_count = totalPrimitives;
	outMesh->vertices = calloc(totalVertices, sizeof(Vertex));
	outMesh->indices = malloc(totalIndices * sizeof(u32));
	outMesh->primitives = calloc(totalPrimitives, sizeof(Primitive));

	uint32_t vertexOffset = 0;
	uint32_t indexOffset = 0;
	uint32_t primitiveIndex = 0;
	mat4 identity;
	glm_mat4_identity(identity);

	for (cgltf_size i = 0; i < data->scenes[0].nodes_count; ++i)
	{
		processNode(data->scenes[0].nodes[i], outMesh, data, identity, NULL, &vertexOffset, &indexOffset, &primitiveIndex);
	}

	// Legacy support - use first material for backward compatibility
	if (outMesh->material_count > 0)
	{
		memcpy(outMesh->base_color, outMesh->materials[0].base_color, sizeof(vec4));
		outMesh->has_texture = outMesh->materials[0].has_texture;
		if (outMesh->materials[0].texture_path)
		{
			outMesh->texture_path = malloc(strlen(outMesh->materials[0].texture_path) + 1);
			strcpy(outMesh->texture_path, outMesh->materials[0].texture_path);
		}
	}
	else
	{
		// Fallback
		outMesh->texture_path = malloc(strlen("Bark_DeadTree.png") + 1);
		strcpy(outMesh->texture_path, "Bark_DeadTree.png");
	}

	free(dir_path);
	cgltf_free(data);
}

void createGrassField(Application* app)
{
	GrassField* grass = &app->grassField;
	memset(grass, 0, sizeof(GrassField));

	// Create grass instances in a grid pattern
	grass->instance_count = 25000; // Much more grass instances
	int gridSize = 158;            // ~158x158 grid approximately
	float spacing = 0.2f;          // Much closer spacing for dense grass

	grass->instances = malloc(grass->instance_count * sizeof(GrassInstance));

	// Generate grass positions with multiple blades per grid cell
	int instanceIndex = 0;
	int bladesPerCell = 3; // Multiple blades per grid position

	for (int x = 0; x < gridSize && instanceIndex < (int)grass->instance_count; x++)
	{
		for (int z = 0; z < gridSize && instanceIndex < (int)grass->instance_count; z++)
		{
			// Add multiple grass blades per grid cell
			for (int blade = 0; blade < bladesPerCell && instanceIndex < (int)grass->instance_count; blade++)
			{
				GrassInstance* instance = &grass->instances[instanceIndex];

				// Base grid position
				float baseX = (x - gridSize / 2) * spacing;
				float baseZ = (z - gridSize / 2) * spacing;

				// Add random offset within the cell
				float randomX = ((float)rand() / RAND_MAX - 0.5f) * spacing * 0.9f;
				float randomZ = ((float)rand() / RAND_MAX - 0.5f) * spacing * 0.9f;

				instance->position[0] = baseX + randomX;
				instance->position[1] = 0.0f; // Ground level
				instance->position[2] = baseZ + randomZ;
				instance->scale = 0.7f + ((float)rand() / RAND_MAX) * 0.6f; // Random scale 0.7-1.3

				instanceIndex++;
			}
		}
	}

	// Create instance buffer
	createBuffer(app, &grass->instanceBuffer,
	    grass->instance_count * sizeof(GrassInstance),
	    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

	// Copy instance data to buffer
	memcpy(grass->instanceBuffer.data, grass->instances,
	    grass->instance_count * sizeof(GrassInstance));

	// Load grass shaders
	grass->vertShaderModule = LoadShaderModule("shaders/grass.vert.spv", app->device);
	grass->fragShaderModule = LoadShaderModule("shaders/grass.frag.spv", app->device);

	// Create grass pipeline layout (reuse main descriptor set layout)
	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
	    .setLayoutCount = 1,
	    .pSetLayouts = &app->descriptorSetLayout,
	    .pushConstantRangeCount = 0,
	};

	VK_CHECK(vkCreatePipelineLayout(app->device, &pipelineLayoutInfo, NULL, &grass->pipelineLayout));

	// Create grass pipeline
	grass->pipeline = createGrassPipeline(app, grass->vertShaderModule, grass->fragShaderModule);
}

void createGroundPlane(Mesh* outMesh)
{
	// Initialize mesh
	memset(outMesh, 0, sizeof(Mesh));

	// Create a large plane (1000x1000 units) centered at origin
	float size = 500.0f;    // Half-size, so total size is 1000x1000
	float texScale = 50.0f; // Texture repetition factor

	// 4 vertices for a quad
	outMesh->vertex_count = 4;
	outMesh->vertices = malloc(4 * sizeof(Vertex));

	// Bottom-left
	outMesh->vertices[0] = (Vertex){
	    .pos = {-size, -0.1f, -size}, // Slightly below Y=0
	    .normal = {0.0f, 1.0f, 0.0f},
	    .texcoord = {0.0f, 0.0f},
	    .color = {1.0f, 1.0f, 1.0f, 1.0f}};

	// Bottom-right
	outMesh->vertices[1] = (Vertex){
	    .pos = {size, -0.1f, -size},
	    .normal = {0.0f, 1.0f, 0.0f},
	    .texcoord = {texScale, 0.0f},
	    .color = {1.0f, 1.0f, 1.0f, 1.0f}};

	// Top-right
	outMesh->vertices[2] = (Vertex){
	    .pos = {size, -0.1f, size},
	    .normal = {0.0f, 1.0f, 0.0f},
	    .texcoord = {texScale, texScale},
	    .color = {1.0f, 1.0f, 1.0f, 1.0f}};

	// Top-left
	outMesh->vertices[3] = (Vertex){
	    .pos = {-size, -0.1f, size},
	    .normal = {0.0f, 1.0f, 0.0f},
	    .texcoord = {0.0f, texScale},
	    .color = {1.0f, 1.0f, 1.0f, 1.0f}};

	// 6 indices for 2 triangles
	outMesh->index_count = 6;
	outMesh->indices = malloc(6 * sizeof(u32));

	// First triangle (bottom-left, bottom-right, top-right)
	outMesh->indices[0] = 0;
	outMesh->indices[1] = 1;
	outMesh->indices[2] = 2;

	// Second triangle (bottom-left, top-right, top-left)
	outMesh->indices[3] = 0;
	outMesh->indices[4] = 2;
	outMesh->indices[5] = 3;

	// Create single material for ground plane
	outMesh->material_count = 1;
	outMesh->materials = calloc(1, sizeof(Material));
	outMesh->materials[0].base_color[0] = 1.0f;
	outMesh->materials[0].base_color[1] = 1.0f;
	outMesh->materials[0].base_color[2] = 1.0f;
	outMesh->materials[0].base_color[3] = 1.0f;
	outMesh->materials[0].has_texture = 1;
	outMesh->materials[0].texture_index = 0;
	outMesh->materials[0].texture_path = malloc(strlen("Bark_DeadTree.png") + 1);
	strcpy(outMesh->materials[0].texture_path, "Bark_DeadTree.png");

	// Create single primitive
	outMesh->primitive_count = 1;
	outMesh->primitives = calloc(1, sizeof(Primitive));
	outMesh->primitives[0].first_index = 0;
	outMesh->primitives[0].index_count = 6;
	outMesh->primitives[0].material_index = 0;

	// Legacy support
	outMesh->texture_path = malloc(strlen("Bark_DeadTree.png") + 1);
	strcpy(outMesh->texture_path, "Bark_DeadTree.png");
	outMesh->has_texture = 1;
	memcpy(outMesh->base_color, outMesh->materials[0].base_color, sizeof(vec4));
}

void renderGrass(Application* app, VkCommandBuffer commandBuffer)
{
	GrassField* grass = &app->grassField;

	// Bind grass pipeline
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, grass->pipeline);

	// Bind descriptor set (reuse main descriptor set)
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	    grass->pipelineLayout, 0, 1, &app->descriptorSet, 0, NULL);

	// Bind instance buffer
	VkBuffer instanceBuffers[] = {grass->instanceBuffer.vkbuffer};
	VkDeviceSize offsets[] = {0};
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, instanceBuffers, offsets);

	// Draw grass instances (6 vertices per blade: 2 triangles * 3 vertices)
	vkCmdDraw(commandBuffer, 6, grass->instance_count, 0, 0);
}

void cleanupGrass(Application* app)
{
	GrassField* grass = &app->grassField;

	if (grass->pipeline != VK_NULL_HANDLE)
	{
		vkDestroyPipeline(app->device, grass->pipeline, NULL);
	}
	if (grass->pipelineLayout != VK_NULL_HANDLE)
	{
		vkDestroyPipelineLayout(app->device, grass->pipelineLayout, NULL);
	}
	if (grass->vertShaderModule != VK_NULL_HANDLE)
	{
		vkDestroyShaderModule(app->device, grass->vertShaderModule, NULL);
	}
	if (grass->fragShaderModule != VK_NULL_HANDLE)
	{
		vkDestroyShaderModule(app->device, grass->fragShaderModule, NULL);
	}

	destroyBuffer(app->device, &grass->instanceBuffer);

	if (grass->instances)
	{
		free(grass->instances);
		grass->instances = NULL;
	}

	memset(grass, 0, sizeof(GrassField));
}

// --- Memory/Buffer Helpers ---

u32 selectmemorytype(
    VkPhysicalDeviceMemoryProperties* memprops, u32 memtypeBits, VkFlags requirements_mask)
{
	for (u32 i = 0; i < memprops->memoryTypeCount; ++i)
	{
		if ((memtypeBits & 1) == 1)
		{
			if ((memprops->memoryTypes[i].propertyFlags & requirements_mask) ==
			    requirements_mask)
			{
				return i;
			}
		}
		memtypeBits >>= 1;
	}
	assert(0 && "No suitable memory type found");
	return 0;
}

void createBuffer(Application* app, Buffer* buffer, VkDeviceSize size, VkBufferUsageFlags usage)
{
	buffer->size = size;

	// 1. Create buffer
	VkBufferCreateInfo bufferInfo = {
	    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
	    .size = size,
	    .usage = usage,
	    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VK_CHECK(vkCreateBuffer(app->device, &bufferInfo, NULL, &buffer->vkbuffer));

	// 2. Get memory requirements
	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(app->device, buffer->vkbuffer, &memRequirements);

	// 3. Find memory type and allocate
	VkMemoryPropertyFlags desiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	uint32_t memoryTypeIndex = findMemoryType(&app->memoryProperties, memRequirements.memoryTypeBits, desiredFlags);
	VkMemoryAllocateInfo allocInfo = {
	    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	    .pNext = NULL,
	    .allocationSize = bufferInfo.size,
	};
	allocInfo.memoryTypeIndex = memoryTypeIndex;

	VK_CHECK(vkAllocateMemory(app->device, &allocInfo, NULL, &buffer->memory));
	VK_CHECK(vkBindBufferMemory(app->device, buffer->vkbuffer, buffer->memory, 0));

	// Now check property flags before mapping
	if (desiredFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
	{
		VK_CHECK(vkMapMemory(app->device, buffer->memory, 0, size, 0, &buffer->data));
	}
}

void destroyBuffer(VkDevice device, Buffer* buffer)
{
	if (buffer->data)
	{
		vkUnmapMemory(device, buffer->memory);
		buffer->data = NULL;
	}
	if (buffer->vkbuffer)
	{
		vkDestroyBuffer(device, buffer->vkbuffer, NULL);
	}
	if (buffer->memory)
	{
		vkFreeMemory(device, buffer->memory, NULL);
	}
}

// --- Vulkan Initialization ---
VkInstance createVulkanInstance(void)
{
	VkApplicationInfo appInfo = {
	    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
	    .pApplicationName = "Vulkan Test",
	    .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
	    .pEngineName = "No Engine",
	    .engineVersion = VK_MAKE_VERSION(1, 0, 0),
	    .apiVersion = VK_API_VERSION_1_0,
	};

	VkInstanceCreateInfo createInfo = {
	    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
	    .pApplicationInfo = &appInfo,
	};

#ifdef _DEBUG
	const char* debugLayers[] = {"VK_LAYER_KHRONOS_validation", " VK_EXT_DEBUG_REPORT_EXTENSION_NAME"};
	createInfo.ppEnabledLayerNames = debugLayers;
	createInfo.enabledLayerCount = ARRAYSIZE(debugLayers);
#endif

	u32 glfwExtensionCount = 0;
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	const char* extensions[16];
	assert(glfwExtensionCount < 15);

	for (u32 i = 0; i < glfwExtensionCount; ++i)
	{
		extensions[i] = glfwExtensions[i];
	}

	u32 extensionCount = glfwExtensionCount;

	createInfo.ppEnabledExtensionNames = extensions;
	createInfo.enabledExtensionCount = extensionCount;

	VkInstance instance;
	VK_CHECK(vkCreateInstance(&createInfo, NULL, &instance));
	return instance;
}

VKAPI_ATTR VkBool32 VKAPI_CALL vulkan_debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
	(void)messageSeverity;
	(void)messageType;
	(void)pUserData;
	fprintf(stderr, "validation layer: %s\n", pCallbackData->pMessage);
	return VK_FALSE;
}

void CreateDebugCallback(Application* app)
{
	VkDebugUtilsMessengerCreateInfoEXT create_info = {
	    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
	    .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
	                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
	                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
	                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
	    .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
	                   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
	                   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
	    .pfnUserCallback = vulkan_debug_callback,
	};

	VK_CHECK(vkCreateDebugUtilsMessengerEXT(app->instance, &create_info, NULL, &app->DebugMessenger));
}

VkPhysicalDevice selectPhysicalDevice(VkInstance instance)
{
	VkPhysicalDevice physicalDevices[8];
	u32 count = ARRAYSIZE(physicalDevices);
	VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, physicalDevices));

	VkPhysicalDevice selected = VK_NULL_HANDLE;

	for (u32 i = 0; i < count; ++i)
	{
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(physicalDevices[i], &props);

		if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
		    props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
		{
			printf("GPU%d: %s (%s)\n", i, props.deviceName,
			    props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "Discrete GPU" : "Integrated GPU");
			printf("  Vulkan API: %d.%d.%d\n",
			    VK_VERSION_MAJOR(props.apiVersion),
			    VK_VERSION_MINOR(props.apiVersion),
			    VK_VERSION_PATCH(props.apiVersion));
			printf("  Driver: %d.%d.%d\n",
			    VK_VERSION_MAJOR(props.driverVersion),
			    VK_VERSION_MINOR(props.driverVersion),
			    VK_VERSION_PATCH(props.driverVersion));

			// Prefer discrete GPU
			if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
				selected = physicalDevices[i];
			else if (!selected)
				selected = physicalDevices[i];
		}
	}

	if (!selected)
	{
		fprintf(stderr, "No suitable GPU found (integrated or discrete only).\n");
		exit(1);
	}

	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(selected, &props);

	printf("\n=== SELECTED GPU ===\n");
	printf("Name: %s\n", props.deviceName);
	printf("Type: %s\n", props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "Discrete GPU" : "Integrated GPU");
	printf("Vendor ID: 0x%X\n", props.vendorID);
	printf("Device ID: 0x%X\n", props.deviceID);
	printf("Vulkan API: %d.%d.%d\n",
	    VK_VERSION_MAJOR(props.apiVersion),
	    VK_VERSION_MINOR(props.apiVersion),
	    VK_VERSION_PATCH(props.apiVersion));
	printf("Driver: %d.%d.%d\n",
	    VK_VERSION_MAJOR(props.driverVersion),
	    VK_VERSION_MINOR(props.driverVersion),
	    VK_VERSION_PATCH(props.driverVersion));
	printf("Max Texture Size: %d x %d\n",
	    props.limits.maxImageDimension2D,
	    props.limits.maxImageDimension2D);
	printf("Max Uniform Buffer Size: %zu MB\n",
	    props.limits.maxUniformBufferRange / (1024 * 1024));
	printf("====================\n\n");

	return selected;
}
u32 find_graphics_queue_family_index(VkPhysicalDevice pickedPhysicalDevice)
{
	u32 queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(pickedPhysicalDevice,
	    &queueFamilyCount, NULL);
	VkQueueFamilyProperties* queueFamilies =
	    malloc(queueFamilyCount * sizeof(VkQueueFamilyProperties));
	vkGetPhysicalDeviceQueueFamilyProperties(pickedPhysicalDevice,
	    &queueFamilyCount, queueFamilies);
	u32 queuefamilyIndex = UINT32_MAX;
	for (u32 i = 0; i < queueFamilyCount; ++i)
	{
		if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
		{
			queuefamilyIndex = i;
			break;
		}
	}
	assert(queuefamilyIndex != UINT32_MAX && "No suitable queue family found");
	free(queueFamilies);
	return queuefamilyIndex;
}

VkDevice create_logical_device(VkPhysicalDevice pickedPhysicaldevice, u32 queueFamilyIndex)
{
	float queuePriority = 1.0f;
	VkDeviceQueueCreateInfo queueCreateInfo = {
	    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
	    .queueFamilyIndex = queueFamilyIndex,
	    .queueCount = 1,
	    .pQueuePriorities = &queuePriority,
	};
	VkPhysicalDeviceFeatures deviceFeatures = {
	    .samplerAnisotropy = VK_TRUE,
	};
	const char* deviceExtensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
	VkDeviceCreateInfo deviceCreateInfo = {
	    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
	    .queueCreateInfoCount = 1,
	    .pQueueCreateInfos = &queueCreateInfo,
	    .enabledExtensionCount = ARRAYSIZE(deviceExtensions),
	    .ppEnabledExtensionNames = deviceExtensions,
	    .pEnabledFeatures = &deviceFeatures,
	};

	VkDevice device;
	VK_CHECK(
	    vkCreateDevice(pickedPhysicaldevice, &deviceCreateInfo, 0, &device));
	return device;
}

VkSurfaceKHR createSurface(VkInstance instance, GLFWwindow* window)
{
	VkSurfaceKHR surface;
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
	VkWaylandSurfaceCreateInfoKHR surfacecreateInfo = {
	    .sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
	    .display = glfwGetWaylandDisplay(),
	    .surface = glfwGetWaylandWindow(window),
	};
	VK_CHECK(vkCreateWaylandSurfaceKHR(instance, &surfacecreateInfo, 0, &surface));
#elif defined(VK_USE_PLATFORM_XLIB_KHR)
	VkXlibSurfaceCreateInfoKHR surfacecreateInfo = {
	    .sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
	    .dpy = glfwGetX11Display(),
	    .window = glfwGetX11Window(window),
	};
	VK_CHECK(vkCreateXlibSurfaceKHR(instance, &surfacecreateInfo, 0, &surface));
#else
	fprintf(stderr, "No supported platform defined for Vulkan surface creation\n");
	exit(1);
#endif
	return surface;
}

VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device)
{
	VkDescriptorSetLayoutBinding bindings[] = {
	    {
	        .binding = 0,
	        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	        .descriptorCount = 1,
	        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	    },
	    {
	        .binding = 1,
	        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
	        .descriptorCount = 1,
	        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
	    },
	    {
	        .binding = 2,
	        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
	        .descriptorCount = 1,
	        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
	    },
	    {
	        .binding = 3,
	        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	        .descriptorCount = 1,
	        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
	    },
	    {
	        .binding = 4,
	        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	        .descriptorCount = 1,
	        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
	    },
	};

	VkDescriptorSetLayoutCreateInfo layoutInfo = {
	    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	    .bindingCount = ARRAYSIZE(bindings),
	    .pBindings = bindings,
	};

	VkDescriptorSetLayout descriptorSetLayout;
	VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, NULL, &descriptorSetLayout));
	return descriptorSetLayout;
}

VkDescriptorSetLayout createGridDescriptorSetLayout(VkDevice device)
{
	VkDescriptorSetLayoutBinding uboLayoutBinding = {
	    .binding = 0,
	    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	    .descriptorCount = 1,
	    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	    .pImmutableSamplers = NULL,
	};

	VkDescriptorSetLayoutBinding samplerLayoutBinding = {
	    .binding = 1,
	    .descriptorCount = 1,
	    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
	    .pImmutableSamplers = NULL,
	    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	VkDescriptorSetLayoutBinding baseColorBinding = {
	    .binding = 3,
	    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	    .descriptorCount = 1,
	    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
	    .pImmutableSamplers = NULL};

	VkDescriptorSetLayoutBinding hasTextureBinding = {
	    .binding = 4,
	    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	    .descriptorCount = 1,
	    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
	    .pImmutableSamplers = NULL};

	// include all 4 bindings
	VkDescriptorSetLayoutBinding bindings[] = {
	    uboLayoutBinding,
	    samplerLayoutBinding,
	    baseColorBinding,
	    hasTextureBinding};

	VkDescriptorSetLayoutCreateInfo layoutInfo = {
	    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	    .bindingCount = ARRAYSIZE(bindings),
	    .pBindings = bindings,

	};

	VkDescriptorSetLayout descriptorSetLayout;
	VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, NULL, &descriptorSetLayout));
	return descriptorSetLayout;
}

VkDescriptorPool createDescriptorPool(VkDevice device)
{
	VkDescriptorPoolSize poolSizes[] = {
	    {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 200},         // Much more for many materials (26 textures × 3 uniform buffers each + extra)
	    {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 200}, // Much more for many textures (26 textures × 2 samplers each + extra)
	};

	VkDescriptorPoolCreateInfo poolInfo = {
	    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
	    .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, // Allow freeing individual sets
	    .maxSets = 100,                                             // Allow many descriptor sets
	    .poolSizeCount = 2,
	    .pPoolSizes = poolSizes,
	};

	VkDescriptorPool descriptorPool;
	VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, NULL, &descriptorPool));
	return descriptorPool;
}

VkDescriptorSet allocateDescriptorSet(VkDevice device, VkDescriptorPool pool, VkDescriptorSetLayout layout)
{
	VkDescriptorSetAllocateInfo allocInfo = {
	    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
	    .descriptorPool = pool,
	    .descriptorSetCount = 1,
	    .pSetLayouts = &layout};

	VkDescriptorSet descriptorSet;
	VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
	return descriptorSet;
}

VkSwapchainKHR createSwapchain(Application* app)
{
	u32 queueFamilyIndex = find_graphics_queue_family_index(app->physicalDevice);
	VkBool32 presentSupported = 0;
	VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(
	    app->physicalDevice, queueFamilyIndex, app->surface, &presentSupported));
	assert(presentSupported);
	VkSurfaceCapabilitiesKHR surfaceCapabilities;
	VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
	    app->physicalDevice, app->surface, &surfaceCapabilities));
	VkSwapchainCreateInfoKHR swapchainInfo = {
	    .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
	    .surface = app->surface,
	    .minImageCount = surfaceCapabilities.minImageCount,
	    .imageFormat = app->swapchainFormat,
	    .imageColorSpace = app->swapchainColorSpace,
	    .imageExtent = {.width = app->width, .height = app->height},
	    .imageArrayLayers = 1,
	    .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
	    .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
	    .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
	    .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
	    .presentMode = VK_PRESENT_MODE_FIFO_KHR,
	    .clipped = VK_TRUE,
	    .queueFamilyIndexCount = 1,
	    .pQueueFamilyIndices = &queueFamilyIndex,
	};
	VkSwapchainKHR swapchain;
	VK_CHECK(vkCreateSwapchainKHR(app->device, &swapchainInfo, 0, &swapchain));
	return swapchain;
}

VkRenderPass createRenderPass(VkDevice device, VkFormat colorFormat, VkFormat inDepthFormat)
{
	VkAttachmentDescription attachments[2] = {
	    {.format = colorFormat, .samples = VK_SAMPLE_COUNT_1_BIT, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
	    {.format = inDepthFormat, .samples = VK_SAMPLE_COUNT_1_BIT, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}};
	VkAttachmentReference colorRef = {.attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
	VkAttachmentReference depthRef = {.attachment = 1, .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
	VkSubpassDescription subpass = {
	    .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
	    .colorAttachmentCount = 1,
	    .pColorAttachments = &colorRef,
	    .pDepthStencilAttachment = &depthRef};
	VkRenderPassCreateInfo rpInfo = {
	    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
	    .attachmentCount = 2,
	    .pAttachments = attachments,
	    .subpassCount = 1,
	    .pSubpasses = &subpass};
	VkRenderPass renderPass;
	VK_CHECK(vkCreateRenderPass(device, &rpInfo, NULL, &renderPass));
	return renderPass;
}
void createSwapchainViews(Application* app)
{
	app->swapchainImageViews = malloc(app->swapchainImageCount * sizeof(VkImageView));
	for (u32 i = 0; i < app->swapchainImageCount; ++i)
	{
		VkImageViewCreateInfo viewInfo = {
		    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		    .image = app->swapchainImages[i],
		    .viewType = VK_IMAGE_VIEW_TYPE_2D,
		    .format = app->swapchainFormat,
		    .components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
		        VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
		    .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1},
		};
		VK_CHECK(vkCreateImageView(app->device, &viewInfo, NULL, &app->swapchainImageViews[i]));
	}
}
void createFramebuffers(Application* app)
{
	app->framebuffers = malloc(app->swapchainImageCount * sizeof(VkFramebuffer));
	for (u32 i = 0; i < app->swapchainImageCount; ++i)
	{
		VkImageView attachments[2] = {app->swapchainImageViews[i], app->depthImageView};
		VkFramebufferCreateInfo fbInfo = {
		    .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		    .renderPass = app->renderPass,
		    .attachmentCount = 2,
		    .pAttachments = attachments,
		    .width = app->width,
		    .height = app->height,
		    .layers = 1};
		VK_CHECK(vkCreateFramebuffer(app->device, &fbInfo, NULL, &app->framebuffers[i]));
	}
}

// --- Vulkan Helpers ---

VkShaderModule LoadShaderModule(const char* filepath, VkDevice device)
{
	FILE* file = fopen(filepath, "rb");
	assert(file);

	fseek(file, 0, SEEK_END);
	long length = ftell(file);
	assert(length >= 0);
	fseek(file, 0, SEEK_SET);

	char* buffer = (char*)malloc(length);
	assert(buffer);

	size_t rc = fread(buffer, 1, length, file);
	assert(rc == (size_t)length);
	fclose(file);

	VkShaderModuleCreateInfo createInfo = {0};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = length;
	createInfo.pCode = (const u32*)buffer;

	VkShaderModule shaderModule;
	VK_CHECK(vkCreateShaderModule(device, &createInfo, NULL, &shaderModule));

	free(buffer);
	return shaderModule;
}

VkFormat findDepthFormat(VkPhysicalDevice physicalDevice)
{
	const VkFormat candidates[] = {
	    VK_FORMAT_D32_SFLOAT,
	    VK_FORMAT_D32_SFLOAT_S8_UINT,
	    VK_FORMAT_D24_UNORM_S8_UINT};

	for (size_t i = 0; i < ARRAYSIZE(candidates); i++)
	{
		VkFormatProperties props;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, candidates[i], &props);

		if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
		{
			return candidates[i];
		}
	}

	assert(0 && "Failed to find supported depth format");
	return VK_FORMAT_UNDEFINED;
}

void createDepthResources(Application* app)
{
	app->depthFormat = findDepthFormat(app->physicalDevice);

	VkImageCreateInfo imageInfo = {
	    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
	    .imageType = VK_IMAGE_TYPE_2D,
	    .format = app->depthFormat,
	    .extent = {app->width, app->height, 1},
	    .mipLevels = 1,
	    .arrayLayers = 1,
	    .samples = VK_SAMPLE_COUNT_1_BIT,
	    .tiling = VK_IMAGE_TILING_OPTIMAL,
	    .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
	    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};

	VK_CHECK(vkCreateImage(app->device, &imageInfo, NULL, &app->depthImage));

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(app->device, app->depthImage, &memRequirements);

	VkMemoryAllocateInfo allocInfo = {
	    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	    .allocationSize = memRequirements.size,
	    .memoryTypeIndex = selectmemorytype(&app->memoryProperties, memRequirements.memoryTypeBits,
	        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};

	VK_CHECK(vkAllocateMemory(app->device, &allocInfo, NULL, &app->depthImageMemory));
	VK_CHECK(vkBindImageMemory(app->device, app->depthImage, app->depthImageMemory, 0));

	VkImageViewCreateInfo viewInfo = {
	    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,

	    .image = app->depthImage,
	    .viewType = VK_IMAGE_VIEW_TYPE_2D,
	    .format = app->depthFormat,

	    .subresourceRange = {
	        .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
	        .baseMipLevel = 0,
	        .levelCount = 1,
	        .baseArrayLayer = 0,
	        .layerCount = 1}};

	VK_CHECK(vkCreateImageView(app->device, &viewInfo, NULL, &app->depthImageView));
}

VkPipeline createGraphicsPipeline(Application* app, VkShaderModule vertShader, VkShaderModule fragShader)
{
	VkPipelineShaderStageCreateInfo stages[2] = {
	    {
	        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	        .stage = VK_SHADER_STAGE_VERTEX_BIT,
	        .module = vertShader,
	        .pName = "main",
	    },
	    {
	        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
	        .module = fragShader,
	        .pName = "main",
	    },
	};

	// Vertex input layout
	VkVertexInputBindingDescription bindingDesc = {
	    .binding = 0,
	    .stride = sizeof(Vertex),
	    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	VkVertexInputAttributeDescription attributes[] = {
	    {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, pos)},
	    {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal)},
	    {.location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, texcoord)},
	    {.location = 3, .binding = 0, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = offsetof(Vertex, color)}, // ✅
	};

	VkPipelineVertexInputStateCreateInfo vertexInput = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
	    .vertexBindingDescriptionCount = 1,
	    .pVertexBindingDescriptions = &bindingDesc,
	    .vertexAttributeDescriptionCount = 4,
	    .pVertexAttributeDescriptions = attributes,
	};

	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
	    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	VkPipelineViewportStateCreateInfo viewportState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
	    .viewportCount = 1,
	    .scissorCount = 1,
	};

	VkPipelineRasterizationStateCreateInfo rasterizationState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
	    .lineWidth = 1.f,
	    .cullMode = VK_CULL_MODE_BACK_BIT,
	    .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
	};

	VkPipelineMultisampleStateCreateInfo multisampleState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
	    .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	VkPipelineDepthStencilStateCreateInfo depthStencilState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
	    .depthTestEnable = VK_TRUE,
	    .depthWriteEnable = VK_TRUE,
	    .depthCompareOp = VK_COMPARE_OP_LESS,
	    .depthBoundsTestEnable = VK_FALSE,
	    .stencilTestEnable = VK_FALSE,
	};

	VkPipelineColorBlendAttachmentState colorBlendAttachment = {
	    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	VkPipelineColorBlendStateCreateInfo colorBlendState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
	    .attachmentCount = 1,
	    .pAttachments = &colorBlendAttachment,
	};

	VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamicState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
	    .dynamicStateCount = ARRAYSIZE(dynamicStates),
	    .pDynamicStates = dynamicStates,
	};

	VkGraphicsPipelineCreateInfo pipelineInfo = {
	    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
	    .stageCount = ARRAYSIZE(stages),
	    .pStages = stages,
	    .pVertexInputState = &vertexInput,
	    .pInputAssemblyState = &inputAssembly,
	    .pViewportState = &viewportState,
	    .pRasterizationState = &rasterizationState,
	    .pMultisampleState = &multisampleState,
	    .pDepthStencilState = &depthStencilState,
	    .pColorBlendState = &colorBlendState,
	    .pDynamicState = &dynamicState,
	    .layout = app->pipelineLayout,
	    .renderPass = app->renderPass,
	};

	VkPipeline pipeline;
	VK_CHECK(vkCreateGraphicsPipelines(app->device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &pipeline));
	return pipeline;
}

VkPipeline createGridPipeline(Application* app, VkShaderModule vertShader, VkShaderModule fragShader)
{
	VkPipelineShaderStageCreateInfo stages[2] = {
	    {
	        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	        .stage = VK_SHADER_STAGE_VERTEX_BIT,
	        .module = vertShader,
	        .pName = "main",
	    },
	    {
	        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
	        .module = fragShader,
	        .pName = "main",
	    },
	};

	// Vertex input for grid - simple 2D positions
	VkVertexInputBindingDescription gridBindingDesc = {
	    .binding = 0,
	    .stride = 2 * sizeof(float), // vec2
	    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	VkVertexInputAttributeDescription gridAttribute = {
	    .location = 0,
	    .binding = 0,
	    .format = VK_FORMAT_R32G32_SFLOAT,
	    .offset = 0};

	VkPipelineVertexInputStateCreateInfo vertexInput = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
	    .vertexBindingDescriptionCount = 1,
	    .pVertexBindingDescriptions = &gridBindingDesc,
	    .vertexAttributeDescriptionCount = 1,
	    .pVertexAttributeDescriptions = &gridAttribute,
	};

	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
	    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	VkPipelineViewportStateCreateInfo viewportState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
	    .viewportCount = 1,
	    .scissorCount = 1,
	};

	VkPipelineRasterizationStateCreateInfo rasterizationState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
	    .lineWidth = 1.f,
	    .cullMode = VK_CULL_MODE_NONE, // Don't cull for fullscreen quad
	    .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
	};

	VkPipelineMultisampleStateCreateInfo multisampleState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
	    .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	VkPipelineDepthStencilStateCreateInfo depthStencilState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
	    .depthTestEnable = VK_TRUE,
	    .depthWriteEnable = VK_TRUE, // Grid should write depth for proper z-testing
	    .depthCompareOp = VK_COMPARE_OP_LESS,
	    .depthBoundsTestEnable = VK_FALSE,
	    .stencilTestEnable = VK_FALSE,
	};

	// Enable blending for grid transparency
	VkPipelineColorBlendAttachmentState colorBlendAttachment = {
	    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	    .blendEnable = VK_TRUE,
	    .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
	    .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
	    .colorBlendOp = VK_BLEND_OP_ADD,
	    .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
	    .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
	    .alphaBlendOp = VK_BLEND_OP_ADD,
	};

	VkPipelineColorBlendStateCreateInfo colorBlendState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
	    .attachmentCount = 1,
	    .pAttachments = &colorBlendAttachment,
	};

	VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamicState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
	    .dynamicStateCount = ARRAYSIZE(dynamicStates),
	    .pDynamicStates = dynamicStates,
	};

	VkGraphicsPipelineCreateInfo pipelineInfo = {
	    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
	    .stageCount = ARRAYSIZE(stages),
	    .pStages = stages,
	    .pVertexInputState = &vertexInput,
	    .pInputAssemblyState = &inputAssembly,
	    .pViewportState = &viewportState,
	    .pRasterizationState = &rasterizationState,
	    .pMultisampleState = &multisampleState,
	    .pDepthStencilState = &depthStencilState,
	    .pColorBlendState = &colorBlendState,
	    .pDynamicState = &dynamicState,
	    .layout = app->pipelineLayout,
	    .renderPass = app->renderPass,
	};

	VkPipeline pipeline;
	VK_CHECK(vkCreateGraphicsPipelines(app->device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &pipeline));
	return pipeline;
}

VkPipeline createGrassPipeline(Application* app, VkShaderModule vertShader, VkShaderModule fragShader)
{
	VkPipelineShaderStageCreateInfo stages[2] = {
	    {
	        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	        .stage = VK_SHADER_STAGE_VERTEX_BIT,
	        .module = vertShader,
	        .pName = "main",
	    },
	    {
	        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
	        .module = fragShader,
	        .pName = "main",
	    },
	};

	// Vertex input for grass instances
	VkVertexInputBindingDescription bindingDescs[1] = {
	    {
	        .binding = 0,
	        .stride = sizeof(GrassInstance),
	        .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE, // Per-instance data
	    }};

	VkVertexInputAttributeDescription attributeDescs[2] = {
	    {
	        .location = 0, // position
	        .binding = 0,
	        .format = VK_FORMAT_R32G32B32_SFLOAT,
	        .offset = offsetof(GrassInstance, position),
	    },
	    {
	        .location = 1, // scale
	        .binding = 0,
	        .format = VK_FORMAT_R32_SFLOAT,
	        .offset = offsetof(GrassInstance, scale),
	    }};

	VkPipelineVertexInputStateCreateInfo vertexInput = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
	    .vertexBindingDescriptionCount = 1,
	    .pVertexBindingDescriptions = bindingDescs,
	    .vertexAttributeDescriptionCount = 2,
	    .pVertexAttributeDescriptions = attributeDescs,
	};

	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
	    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	VkPipelineViewportStateCreateInfo viewportState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
	    .viewportCount = 1,
	    .scissorCount = 1,
	};

	VkPipelineRasterizationStateCreateInfo rasterizationState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
	    .lineWidth = 1.f,
	    .cullMode = VK_CULL_MODE_NONE, // Don't cull grass
	    .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
	};

	VkPipelineMultisampleStateCreateInfo multisampleState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
	    .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	VkPipelineDepthStencilStateCreateInfo depthStencilState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
	    .depthTestEnable = VK_TRUE,
	    .depthWriteEnable = VK_TRUE,
	    .depthCompareOp = VK_COMPARE_OP_LESS,
	    .depthBoundsTestEnable = VK_FALSE,
	    .stencilTestEnable = VK_FALSE,
	};

	VkPipelineColorBlendAttachmentState colorBlendAttachment = {
	    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	    .blendEnable = VK_FALSE,
	};

	VkPipelineColorBlendStateCreateInfo colorBlendState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
	    .attachmentCount = 1,
	    .pAttachments = &colorBlendAttachment,
	};

	VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamicState = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
	    .dynamicStateCount = ARRAYSIZE(dynamicStates),
	    .pDynamicStates = dynamicStates,
	};

	VkGraphicsPipelineCreateInfo pipelineInfo = {
	    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
	    .stageCount = ARRAYSIZE(stages),
	    .pStages = stages,
	    .pVertexInputState = &vertexInput,
	    .pInputAssemblyState = &inputAssembly,
	    .pViewportState = &viewportState,
	    .pRasterizationState = &rasterizationState,
	    .pMultisampleState = &multisampleState,
	    .pDepthStencilState = &depthStencilState,
	    .pColorBlendState = &colorBlendState,
	    .pDynamicState = &dynamicState,
	    .layout = app->grassField.pipelineLayout,
	    .renderPass = app->renderPass,
	};

	VkPipeline pipeline;
	VK_CHECK(vkCreateGraphicsPipelines(app->device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &pipeline));
	return pipeline;
}

void createSwapchainRelatedResources(Application* app)
{
	// Create swapchain
	app->swapchainFormat = VK_FORMAT_B8G8R8A8_SRGB;
	app->swapchainColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	app->swapchain = createSwapchain(app);

	// Get swapchain images
	VK_CHECK(vkGetSwapchainImagesKHR(app->device, app->swapchain, &app->swapchainImageCount, NULL));
	app->swapchainImages = malloc(app->swapchainImageCount * sizeof(VkImage));
	VK_CHECK(vkGetSwapchainImagesKHR(app->device, app->swapchain, &app->swapchainImageCount, app->swapchainImages));
	createSwapchainViews(app);

	// Create depth resources
	createDepthResources(app);

	// Create render pass
	app->renderPass = createRenderPass(app->device, app->swapchainFormat, app->depthFormat);

	// Create framebuffers
	createFramebuffers(app);
}

void createModelAndBuffers(Application* app)
{
	// Load model - COMMENTED OUT FOR GRASS TESTING
	// loadGltfModel("/home/lka/myprojects/vulkantest3/sponza/Sponza.gltf", &app->mesh);

	// Create ground plane instead
	createGroundPlane(&app->mesh);

	// Create grass field
	createGrassField(app);

	// Create vertex buffer
	size_t vertexBufferSize = app->mesh.vertex_count * sizeof(Vertex);
	Buffer vertexStaging;
	createBuffer(app, &vertexStaging, vertexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	memcpy(vertexStaging.data, app->mesh.vertices, vertexBufferSize);

	createBuffer(app, &app->vertexBuffer, vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

	// Create index buffer
	size_t indexBufferSize = app->mesh.index_count * sizeof(u32);
	Buffer indexStaging;
	createBuffer(app, &indexStaging, indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	memcpy(indexStaging.data, app->mesh.indices, indexBufferSize);

	createBuffer(app, &app->indexBuffer, indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

	// Copy staging buffers to device local buffers
	VkCommandBufferAllocateInfo cmdAllocInfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = app->commandPool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
	VkCommandBuffer copyCmd;
	VK_CHECK(vkAllocateCommandBuffers(app->device, &cmdAllocInfo, &copyCmd));
	VkCommandBufferBeginInfo beginInfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
	VK_CHECK(vkBeginCommandBuffer(copyCmd, &beginInfo));
	VkBufferCopy copyRegion = {.size = vertexBufferSize};
	vkCmdCopyBuffer(copyCmd, vertexStaging.vkbuffer, app->vertexBuffer.vkbuffer, 1, &copyRegion);
	copyRegion.size = indexBufferSize;
	vkCmdCopyBuffer(copyCmd, indexStaging.vkbuffer, app->indexBuffer.vkbuffer, 1, &copyRegion);
	VK_CHECK(vkEndCommandBuffer(copyCmd));
	VkSubmitInfo submitInfo = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &copyCmd};
	VK_CHECK(vkQueueSubmit(app->graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE));
	VK_CHECK(vkQueueWaitIdle(app->graphicsQueue));
	vkFreeCommandBuffers(app->device, app->commandPool, 1, &copyCmd);

	destroyBuffer(app->device, &vertexStaging);
	destroyBuffer(app->device, &indexStaging);

	// Create grid vertex buffer (fullscreen triangle)
	float gridVertices[] = {
	    -1.0f, -1.0f, // Bottom-left
	    3.0f, -1.0f,  // Bottom-right (extended)
	    -1.0f, 3.0f   // Top-left (extended)
	};
	size_t gridVertexBufferSize = sizeof(gridVertices);
	Buffer gridStaging;
	createBuffer(app, &gridStaging, gridVertexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	memcpy(gridStaging.data, gridVertices, gridVertexBufferSize);

	createBuffer(app, &app->gridVertexBuffer, gridVertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

	// Copy grid staging buffer to device local buffer
	VkCommandBuffer gridCopyCmd;
	VK_CHECK(vkAllocateCommandBuffers(app->device, &cmdAllocInfo, &gridCopyCmd));
	VK_CHECK(vkBeginCommandBuffer(gridCopyCmd, &beginInfo));
	VkBufferCopy gridCopyRegion = {.size = gridVertexBufferSize};
	vkCmdCopyBuffer(gridCopyCmd, gridStaging.vkbuffer, app->gridVertexBuffer.vkbuffer, 1, &gridCopyRegion);
	VK_CHECK(vkEndCommandBuffer(gridCopyCmd));
	VkSubmitInfo gridSubmitInfo = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &gridCopyCmd};
	VK_CHECK(vkQueueSubmit(app->graphicsQueue, 1, &gridSubmitInfo, VK_NULL_HANDLE));
	VK_CHECK(vkQueueWaitIdle(app->graphicsQueue));
	vkFreeCommandBuffers(app->device, app->commandPool, 1, &gridCopyCmd);

	destroyBuffer(app->device, &gridStaging);
}

void createTextureResources(Application* app)
{
	// Count unique textures from materials
	app->texture_count = 0;
	for (u32 i = 0; i < app->mesh.material_count; i++)
	{
		if (app->mesh.materials[i].has_texture)
		{
			app->texture_count++;
		}
	}

	// Always have at least one texture (fallback)
	if (app->texture_count == 0)
	{
		app->texture_count = 1;
	}

	// Allocate texture array
	app->textures = calloc(app->texture_count, sizeof(Texture));

	// Load textures and assign indices to materials
	u32 textureIndex = 0;
	for (u32 i = 0; i < app->mesh.material_count; i++)
	{
		if (app->mesh.materials[i].has_texture && app->mesh.materials[i].texture_path)
		{
			u32 mipLevels;
			createTextureImage(app, app->mesh.materials[i].texture_path, &app->textures[textureIndex], &mipLevels);
			createTextureSampler(app, &app->textures[textureIndex], mipLevels);
			app->mesh.materials[i].texture_index = textureIndex;
			textureIndex++;
		}
	}

	// Fallback texture if no materials have textures
	if (textureIndex == 0)
	{
		u32 mipLevels;
		createTextureImage(app, app->mesh.texture_path ? app->mesh.texture_path : "Bark_DeadTree.png", &app->textures[0], &mipLevels);
		createTextureSampler(app, &app->textures[0], mipLevels);
	}

	// Legacy single texture support
	if (app->texture_count > 0)
	{
		app->texture = app->textures[0];
		app->mipLevels = 1; // Default mip levels for legacy support
	}
}

void createUniformBuffer(Application* app)
{
	createBuffer(app, &app->uniformBuffer, sizeof(UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	createBuffer(app, &app->baseColorBuffer, sizeof(vec4), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	createBuffer(app, &app->hasTextureBuffer, sizeof(int), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
}

void createDescriptors(Application* app)
{
	// Create descriptor pool
	app->descriptorPool = createDescriptorPool(app->device);

	// Allocate descriptor sets for each texture
	app->descriptorSets = calloc(app->texture_count, sizeof(VkDescriptorSet));

	for (u32 i = 0; i < app->texture_count; i++)
	{
		app->descriptorSets[i] = allocateDescriptorSet(app->device, app->descriptorPool, app->descriptorSetLayout);

		// Update descriptor set for this texture
		VkDescriptorBufferInfo bufferInfo = {
		    .buffer = app->uniformBuffer.vkbuffer,
		    .offset = 0,
		    .range = sizeof(UniformBufferObject)};

		VkDescriptorImageInfo imageInfo = {
		    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		    .imageView = app->textures[i].view,
		    .sampler = app->textures[i].sampler};

		VkDescriptorImageInfo gridImageInfo = {
		    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		    .imageView = app->gridTexture.view,
		    .sampler = app->gridTexture.sampler};

		VkDescriptorBufferInfo baseColorBufferInfo = {
		    .buffer = app->baseColorBuffer.vkbuffer,
		    .offset = 0,
		    .range = sizeof(vec4)};

		VkDescriptorBufferInfo hasTextureBufferInfo = {
		    .buffer = app->hasTextureBuffer.vkbuffer,
		    .offset = 0,
		    .range = sizeof(int)};

		VkWriteDescriptorSet descriptorWrites[] = {
		    {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = app->descriptorSets[i], .dstBinding = 0, .dstArrayElement = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .pBufferInfo = &bufferInfo},
		    {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = app->descriptorSets[i], .dstBinding = 1, .dstArrayElement = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .pImageInfo = &imageInfo},
		    {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = app->descriptorSets[i], .dstBinding = 2, .dstArrayElement = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .pImageInfo = &gridImageInfo},
		    {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = app->descriptorSets[i], .dstBinding = 3, .dstArrayElement = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .pBufferInfo = &baseColorBufferInfo},
		    {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = app->descriptorSets[i], .dstBinding = 4, .dstArrayElement = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .pBufferInfo = &hasTextureBufferInfo}};

		vkUpdateDescriptorSets(app->device, 5, descriptorWrites, 0, NULL);
	}

	// Legacy single descriptor set support (use first texture)
	app->descriptorSet = app->descriptorSets[0];
}

void createSyncObjects(Application* app)
{
	// // Allocate imageAvailable semaphores per frame in flight
	// app->imageAvailableSemaphores = malloc(MAX_FRAMES_IN_FLIGHT * sizeof(VkSemaphore));
	// // Allocate renderFinished semaphores per swapchain image
	// app->renderFinishedSemaphores = malloc(app->swapchainImageCount * sizeof(VkSemaphore));
	// // Allocate fences per frame in flight
	// app->inFlightFences = malloc(MAX_FRAMES_IN_FLIGHT * sizeof(VkFence));

	VkSemaphoreCreateInfo semaphoreInfo = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
	VkFenceCreateInfo fenceInfo = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT};

	// Create imageAvailable semaphores (per frame in flight)
	for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		VK_CHECK(vkCreateSemaphore(app->device, &semaphoreInfo, NULL, &app->imageAvailableSemaphores[i]));
	}

	// Create renderFinished semaphores (per swapchain image)
	for (u32 i = 0; i < app->swapchainImageCount; i++)
	{
		VK_CHECK(vkCreateSemaphore(app->device, &semaphoreInfo, NULL, &app->renderFinishedSemaphores[i]));
	}

	// Create fences for each frame in flight
	for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		VK_CHECK(vkCreateFence(app->device, &fenceInfo, NULL, &app->inFlightFences[i]));
	}
}

// --- Vulkan Initialization Helpers ---

void createCommandPool(VkDevice device, u32 queueFamilyIndex, VkCommandPool* outCommandPool)
{
	// Create command pool
	VkCommandPoolCreateInfo commandPoolInfo = {
	    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
	    .queueFamilyIndex = queueFamilyIndex,
	    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
	};
	VK_CHECK(vkCreateCommandPool(device, &commandPoolInfo, NULL, outCommandPool));
}

void allocateFrameCommandBuffers(VkDevice device, VkCommandPool commandPool, VkCommandBuffer** outCommandBuffers)
{
	// Allocate command buffers (one per frame in flight)
	*outCommandBuffers = malloc(MAX_FRAMES_IN_FLIGHT * sizeof(VkCommandBuffer));
	VkCommandBufferAllocateInfo cmdAllocInfo = {
	    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
	    .commandPool = commandPool,
	    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
	    .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
	};
	VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, *outCommandBuffers));
}

void createPipeline(Application* app)
{
	// Create descriptor set layout
	app->descriptorSetLayout = createDescriptorSetLayout(app->device);

	// Create pipeline layout
	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
	    .setLayoutCount = 1,
	    .pSetLayouts = &app->descriptorSetLayout,
	};
	VK_CHECK(vkCreatePipelineLayout(app->device, &pipelineLayoutInfo, NULL, &app->pipelineLayout));

	// Load shaders and create main pipeline - USING GRASS SHADERS
	app->vertShaderModule = LoadShaderModule("shaders/grass.vert.spv", app->device);
	app->fragShaderModule = LoadShaderModule("shaders/grass.frag.spv", app->device);
	app->pipeline = createGraphicsPipeline(app, app->vertShaderModule, app->fragShaderModule);

	// Create grid descriptor set layout and pipeline layout
	app->gridDescriptorSetLayout = createGridDescriptorSetLayout(app->device);
	VkPipelineLayoutCreateInfo gridPipelineLayoutInfo = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
	    .setLayoutCount = 1,
	    .pSetLayouts = &app->gridDescriptorSetLayout,
	};
	VK_CHECK(vkCreatePipelineLayout(app->device, &gridPipelineLayoutInfo, NULL, &app->gridPipelineLayout));

	// Load grid shaders and create grid pipeline
	app->gridVertShaderModule = LoadShaderModule("shaders/grid.vert.spv", app->device);
	app->gridFragShaderModule = LoadShaderModule("shaders/grid.frag.spv", app->device);
	app->gridPipeline = createGridPipeline(app, app->gridVertShaderModule, app->gridFragShaderModule);
}

void createResources(Application* app)
{
	createModelAndBuffers(app);
	createTextureResources(app);

	// Load grid texture
	createTextureImage(app, "data/ground.jpg", &app->gridTexture, &app->mipLevels);
	createTextureSampler(app, &app->gridTexture, app->mipLevels);

	createUniformBuffer(app);
	updateBaseColorAndHasTexture(app);

	createDescriptors(app);
}

// --- Vulkan Cleanup Helpers ---

void cleanupSyncObjects(Application* app)
{
	// Clean up imageAvailable semaphores (per frame in flight)
	for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vkDestroySemaphore(app->device, app->imageAvailableSemaphores[i], NULL);
	}
	// Clean up renderFinished semaphores (per swapchain image)
	for (u32 i = 0; i < app->swapchainImageCount; i++)
	{
		vkDestroySemaphore(app->device, app->renderFinishedSemaphores[i], NULL);
	}
	// Clean up fences (per frame in flight)
	for (u32 i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vkDestroyFence(app->device, app->inFlightFences[i], NULL);
	}
	// free(app->renderFinishedSemaphores);
	// free(app->imageAvailableSemaphores);
	// free(app->inFlightFences);
}

void cleanupResources(Application* app)
{
	destroyBuffer(app->device, &app->uniformBuffer);
	destroyBuffer(app->device, &app->indexBuffer);
	destroyBuffer(app->device, &app->vertexBuffer);
	destroyBuffer(app->device, &app->gridVertexBuffer);
	destroyBuffer(app->device, &app->baseColorBuffer);
	destroyBuffer(app->device, &app->hasTextureBuffer);

	// Clean up multiple textures
	if (app->textures)
	{
		for (u32 i = 0; i < app->texture_count; i++)
		{
			vkDestroySampler(app->device, app->textures[i].sampler, NULL);
			vkDestroyImageView(app->device, app->textures[i].view, NULL);
			vkDestroyImage(app->device, app->textures[i].image, NULL);
			vkFreeMemory(app->device, app->textures[i].memory, NULL);
		}
		free(app->textures);
	}

	// Legacy single texture cleanup (in case it's different from textures array)
	if (app->texture.sampler && app->texture.sampler != VK_NULL_HANDLE)
	{
		vkDestroySampler(app->device, app->texture.sampler, NULL);
		vkDestroyImageView(app->device, app->texture.view, NULL);
		vkDestroyImage(app->device, app->texture.image, NULL);
		vkFreeMemory(app->device, app->texture.memory, NULL);
	}

	// Clean up grid texture
	vkDestroySampler(app->device, app->gridTexture.sampler, NULL);
	vkDestroyImageView(app->device, app->gridTexture.view, NULL);
	vkDestroyImage(app->device, app->gridTexture.image, NULL);
	vkFreeMemory(app->device, app->gridTexture.memory, NULL);

	vkDestroyDescriptorPool(app->device, app->descriptorPool, NULL);

	// Clean up descriptor sets array
	if (app->descriptorSets)
	{
		free(app->descriptorSets);
	}

	// Clean up mesh data
	free(app->mesh.vertices);
	free(app->mesh.indices);
	free(app->mesh.texture_path);

	// Clean up materials
	if (app->mesh.materials)
	{
		for (u32 i = 0; i < app->mesh.material_count; i++)
		{
			free(app->mesh.materials[i].texture_path);
		}
		free(app->mesh.materials);
	}

	// Clean up primitives
	free(app->mesh.primitives);
}

void cleanupPipeline(Application* app)
{
	vkDestroyPipeline(app->device, app->pipeline, NULL);
	vkDestroyPipeline(app->device, app->gridPipeline, NULL);
	vkDestroyPipelineLayout(app->device, app->pipelineLayout, NULL);
	vkDestroyShaderModule(app->device, app->fragShaderModule, NULL);
	vkDestroyShaderModule(app->device, app->vertShaderModule, NULL);
	vkDestroyShaderModule(app->device, app->gridFragShaderModule, NULL);
	vkDestroyShaderModule(app->device, app->gridVertShaderModule, NULL);
	vkDestroyDescriptorSetLayout(app->device, app->descriptorSetLayout, NULL);
}

// --- Main Application ---

void framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
	(void)width;
	(void)height;
	Application* app = glfwGetWindowUserPointer(window);
	app->framebufferResized = true;
}

void initWindow(Application* app)
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	app->width = 1280;
	app->height = 720;
	app->window = glfwCreateWindow(app->width, app->height, "Vulkan Test", 0, 0);
	glfwSetWindowUserPointer(app->window, app);
	glfwSetFramebufferSizeCallback(app->window, framebufferResizeCallback);
	glfwSetCursorPosCallback(app->window, mouse_callback);
	glfwSetInputMode(app->window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// Initialize camera - positioned to see the ground plane
	glm_vec3_copy((vec3){0.0f, 5.0f, 10.0f}, app->cameraPos); // Higher up and back
	glm_vec3_copy((vec3){0.0f, 0.0f, -1.0f}, app->cameraFront);
	glm_vec3_copy((vec3){0.0f, 1.0f, 0.0f}, app->cameraUp);
	app->yaw = -90.0f;
	app->pitch = -20.0f; // Look down slightly to see the ground
	app->lastX = app->width / 2.0f;
	app->lastY = app->height / 2.0f;
	app->firstMouse = true;
	app->deltaTime = 0.0f;
	app->lastFrame = 0.0f;

	// Initialize point lights
	app->numActiveLights = 4;

	// Light 1 - Red light
	glm_vec3_copy((vec3){2.0f, 1.0f, 0.0f}, app->lights[0].position);
	glm_vec3_copy((vec3){1.0f, 0.3f, 0.3f}, app->lights[0].color);
	app->lights[0].intensity = 1.0f;
	app->lights[0].constant = 1.0f;
	app->lights[0].linear = 0.09f;
	app->lights[0].quadratic = 0.032f;

	// Light 2 - Green light
	glm_vec3_copy((vec3){-2.0f, 1.0f, 0.0f}, app->lights[1].position);
	glm_vec3_copy((vec3){0.3f, 1.0f, 0.3f}, app->lights[1].color);
	app->lights[1].intensity = 1.0f;
	app->lights[1].constant = 1.0f;
	app->lights[1].linear = 0.09f;
	app->lights[1].quadratic = 0.032f;

	// Light 3 - Blue light
	glm_vec3_copy((vec3){0.0f, 1.0f, 2.0f}, app->lights[2].position);
	glm_vec3_copy((vec3){0.3f, 0.3f, 1.0f}, app->lights[2].color);
	app->lights[2].intensity = 1.0f;
	app->lights[2].constant = 1.0f;
	app->lights[2].linear = 0.09f;
	app->lights[2].quadratic = 0.032f;

	// Light 4 - White light
	glm_vec3_copy((vec3){0.0f, 1.0f, -2.0f}, app->lights[3].position);
	glm_vec3_copy((vec3){1.0f, 1.0f, 1.0f}, app->lights[3].color);
	app->lights[3].intensity = 1.0f;
	app->lights[3].constant = 1.0f;
	app->lights[3].linear = 0.09f;
	app->lights[3].quadratic = 0.032f;
}

void initVulkan(Application* app)
{
	VK_CHECK(volkInitialize());

	// Initialize frame count first
	app->currentFrame = 0;

	// Create instance
	app->instance = createVulkanInstance();
	volkLoadInstance(app->instance);

	// Select physical device
	app->physicalDevice = selectPhysicalDevice(app->instance);
	vkGetPhysicalDeviceMemoryProperties(app->physicalDevice, &app->memoryProperties);

	// Create logical device and queue
	u32 queueFamilyIndex = find_graphics_queue_family_index(app->physicalDevice);
	app->device = create_logical_device(app->physicalDevice, queueFamilyIndex);
	volkLoadDevice(app->device);
	vkGetDeviceQueue(app->device, queueFamilyIndex, 0, &app->graphicsQueue);

	createCommandPool(app->device, queueFamilyIndex, &app->commandPool);
	allocateFrameCommandBuffers(app->device, app->commandPool, &app->commandBuffers);

	// Create surface
	app->surface = createSurface(app->instance, app->window);

	createSwapchainRelatedResources(app);
	createPipeline(app);
	createResources(app);
	createSyncObjects(app);
}

void recordCommandBuffer(Application* app, VkCommandBuffer commandBuffer, u32 imageIndex)
{
	VkCommandBufferBeginInfo beginInfo = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
	VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

	VkClearValue clearValues[2];
	clearValues[0].color = (VkClearColorValue){{0.0f, 0.0f, 0.0f, 1.0f}};
	clearValues[1].depthStencil = (VkClearDepthStencilValue){1.0f, 0};

	VkRenderPassBeginInfo renderPassInfo = {
	    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
	    .renderPass = app->renderPass,
	    .framebuffer = app->framebuffers[imageIndex],
	    .renderArea.offset = {0, 0},
	    .renderArea.extent = {app->width, app->height},
	    .clearValueCount = ARRAYSIZE(clearValues),
	    .pClearValues = clearValues,
	};
	vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->pipeline);

	VkViewport viewport = {.x = 0.0f, .y = 0.0f, .width = (float)app->width, .height = (float)app->height, .minDepth = 0.0f, .maxDepth = 1.0f};
	vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

	VkRect2D scissor = {.offset = {0, 0}, .extent = {app->width, app->height}};
	vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

	VkBuffer vertexBuffers[] = {app->vertexBuffer.vkbuffer};
	VkDeviceSize offsets[] = {0};
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
	vkCmdBindIndexBuffer(commandBuffer, app->indexBuffer.vkbuffer, 0, VK_INDEX_TYPE_UINT32);

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->pipelineLayout, 0, 1, &app->descriptorSet, 0, NULL);

	// Update uniform buffer
	UniformBufferObject ubo = {0};

	// Matrices
	glm_perspective(glm_rad(45.0f), app->width / (float)app->height, 0.1f, 100.0f, ubo.proj);
	ubo.proj[1][1] *= -1; // GLM is for OpenGL, Vulkan Y coordinate is flipped

	vec3 center;
	glm_vec3_add(app->cameraPos, app->cameraFront, center);
	glm_lookat(app->cameraPos, center, app->cameraUp, ubo.view);

	glm_mat4_identity(ubo.model);

	// Camera position and lighting
	glm_vec3_copy(app->cameraPos, ubo.cameraPos);
	ubo.numLights = app->numActiveLights;

	// Copy light data
	for (u32 i = 0; i < app->numActiveLights; i++)
	{
		ubo.lights[i] = app->lights[i];
	}

	memcpy(app->uniformBuffer.data, &ubo, sizeof(UniformBufferObject));

	// Draw grid first (background)
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->gridPipeline);
	VkBuffer gridVertexBuffers[] = {app->gridVertexBuffer.vkbuffer};
	VkDeviceSize gridOffsets[] = {0};
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, gridVertexBuffers, gridOffsets);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->pipelineLayout, 0, 1, &app->descriptorSet, 0, NULL);
	vkCmdDraw(commandBuffer, 3, 1, 0, 0); // Fullscreen triangle

	// Switch back to main pipeline and draw model primitives
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->pipeline);
	VkBuffer vertexBuffers2[] = {app->vertexBuffer.vkbuffer};
	VkDeviceSize offsets2[] = {0};
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers2, offsets2);
	vkCmdBindIndexBuffer(commandBuffer, app->indexBuffer.vkbuffer, 0, VK_INDEX_TYPE_UINT32);

	// Draw each primitive with its material
	for (u32 i = 0; i < app->mesh.primitive_count; i++)
	{
		Primitive* prim = &app->mesh.primitives[i];

		// Update material-specific uniform buffers
		if (prim->material_index >= 0 && prim->material_index < (int)app->mesh.material_count)
		{
			Material* mat = &app->mesh.materials[prim->material_index];

			// Update base color buffer
			memcpy(app->baseColorBuffer.data, mat->base_color, sizeof(vec4));

			// Update has texture buffer
			int hasTexture = mat->has_texture;
			memcpy(app->hasTextureBuffer.data, &hasTexture, sizeof(int));

			// Bind the correct descriptor set for this material's texture
			VkDescriptorSet descriptorSet = app->descriptorSet; // Default
			if (mat->has_texture && mat->texture_index >= 0 && mat->texture_index < (int)app->texture_count)
			{
				descriptorSet = app->descriptorSets[mat->texture_index];
			}
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
		}
		else
		{
			// Use default material
			vec4 defaultColor = {1.0f, 1.0f, 1.0f, 1.0f};
			int hasTexture = 0;
			memcpy(app->baseColorBuffer.data, defaultColor, sizeof(vec4));
			memcpy(app->hasTextureBuffer.data, &hasTexture, sizeof(int));
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->pipelineLayout, 0, 1, &app->descriptorSet, 0, NULL);
		}

		// Draw this primitive
		vkCmdDrawIndexed(commandBuffer, prim->index_count, 1, prim->first_index, 0, 0);
	}

	// Fallback: if no primitives, draw the whole mesh (legacy support)
	if (app->mesh.primitive_count == 0)
	{
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app->pipelineLayout, 0, 1, &app->descriptorSet, 0, NULL);
		vkCmdDrawIndexed(commandBuffer, app->mesh.index_count, 1, 0, 0, 0);
	}

	// Render grass
	renderGrass(app, commandBuffer);

	vkCmdEndRenderPass(commandBuffer);
	VK_CHECK(vkEndCommandBuffer(commandBuffer));
}

void drawFrame(Application* app)
{
	VK_CHECK(vkWaitForFences(app->device, 1, &app->inFlightFences[app->currentFrame], VK_TRUE, UINT64_MAX));

	u32 imageIndex;
	VkResult result = vkAcquireNextImageKHR(app->device, app->swapchain, UINT64_MAX, app->imageAvailableSemaphores[app->currentFrame], VK_NULL_HANDLE, &imageIndex);

	if (result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		recreateSwapchain(app);
		return;
	}
	else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
	{
		assert(0 && "failed to acquire swap chain image!");
	}

	VK_CHECK(vkResetFences(app->device, 1, &app->inFlightFences[app->currentFrame]));

	VkCommandBuffer commandBuffer = app->commandBuffers[app->currentFrame];
	VK_CHECK(vkResetCommandBuffer(commandBuffer, 0));
	recordCommandBuffer(app, commandBuffer, imageIndex);

	VkSubmitInfo submitInfo = {
	    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
	    .waitSemaphoreCount = 1,
	    .pWaitSemaphores = &app->imageAvailableSemaphores[app->currentFrame],
	    .pWaitDstStageMask = (VkPipelineStageFlags[]){VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
	    .commandBufferCount = 1,
	    .pCommandBuffers = &commandBuffer,
	    .signalSemaphoreCount = 1,
	    .pSignalSemaphores = &app->renderFinishedSemaphores[app->currentFrame],
	};
	VK_CHECK(vkQueueSubmit(app->graphicsQueue, 1, &submitInfo, app->inFlightFences[app->currentFrame]));

	VkPresentInfoKHR presentInfo = {
	    .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
	    .waitSemaphoreCount = 1,
	    .pWaitSemaphores = &app->renderFinishedSemaphores[app->currentFrame],
	    .swapchainCount = 1,
	    .pSwapchains = &app->swapchain,
	    .pImageIndices = &imageIndex,
	};
	result = vkQueuePresentKHR(app->graphicsQueue, &presentInfo);

	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || app->framebufferResized)
	{
		app->framebufferResized = false;
		recreateSwapchain(app);
	}
	else if (result != VK_SUCCESS)
	{
		assert(0 && "failed to present swap chain image!");
	}

	app->currentFrame = (app->currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void mainLoop(Application* app)
{
	while (!glfwWindowShouldClose(app->window))
	{

		float currentFrame = glfwGetTime();
		app->deltaTime = currentFrame - app->lastFrame;
		app->lastFrame = currentFrame;

		glfwPollEvents();
		processInput(app);
		updateLights(app);
		drawFrame(app);
	}

	vkDeviceWaitIdle(app->device);
}

void cleanupSwapchain(Application* app)
{
	vkDestroyImageView(app->device, app->depthImageView, NULL);
	vkDestroyImage(app->device, app->depthImage, NULL);
	vkFreeMemory(app->device, app->depthImageMemory, NULL);

	for (u32 i = 0; i < app->swapchainImageCount; i++)
	{
		vkDestroyFramebuffer(app->device, app->framebuffers[i], NULL);
	}
	free(app->framebuffers);

	for (u32 i = 0; i < app->swapchainImageCount; i++)
	{
		vkDestroyImageView(app->device, app->swapchainImageViews[i], NULL);
	}
	free(app->swapchainImageViews);
	free(app->swapchainImages);

	vkDestroyRenderPass(app->device, app->renderPass, NULL);
	vkDestroySwapchainKHR(app->device, app->swapchain, NULL);
}

void recreateSwapchain(Application* app)
{
	int width = 0, height = 0;
	glfwGetFramebufferSize(app->window, &width, &height);
	while (width == 0 || height == 0)
	{
		glfwGetFramebufferSize(app->window, &width, &height);
		glfwWaitEvents();
	}

	vkDeviceWaitIdle(app->device);

	// Destroy pipeline before swapchain resources (like render pass)
	vkDestroyPipeline(app->device, app->pipeline, NULL);
	cleanupSwapchain(app);

	app->width = width;
	app->height = height;

	createSwapchainRelatedResources(app);
	app->pipeline = createGraphicsPipeline(app, app->vertShaderModule, app->fragShaderModule);
}

void cleanup(Application* app)
{
	vkDeviceWaitIdle(app->device);

	cleanupGrass(app);
	cleanupSyncObjects(app);
	cleanupResources(app);
	cleanupPipeline(app);
	cleanupSwapchain(app);

	vkDestroySurfaceKHR(app->instance, app->surface, NULL);
	free(app->commandBuffers);
	vkDestroyCommandPool(app->device, app->commandPool, NULL);
	vkDestroyDevice(app->device, NULL);
	vkDestroyInstance(app->instance, NULL);

	glfwDestroyWindow(app->window);
	glfwTerminate();
}

int main(void)
{
	Application app = {0};
	initWindow(&app);
	initVulkan(&app);
	mainLoop(&app);
	cleanup(&app);
	return 0;
}

// --- Input Handling ---
void processInput(Application* app)
{
	float cameraSpeed = 2.5f * app->deltaTime;
	if (glfwGetKey(app->window, GLFW_KEY_W) == GLFW_PRESS)
	{
		vec3 front;
		glm_vec3_scale(app->cameraFront, cameraSpeed, front);
		glm_vec3_add(app->cameraPos, front, app->cameraPos);
	}
	if (glfwGetKey(app->window, GLFW_KEY_S) == GLFW_PRESS)
	{
		vec3 front;
		glm_vec3_scale(app->cameraFront, cameraSpeed, front);
		glm_vec3_sub(app->cameraPos, front, app->cameraPos);
	}
	if (glfwGetKey(app->window, GLFW_KEY_A) == GLFW_PRESS)
	{
		vec3 right;
		glm_cross(app->cameraFront, app->cameraUp, right);
		glm_normalize(right);
		glm_vec3_scale(right, cameraSpeed, right);
		glm_vec3_sub(app->cameraPos, right, app->cameraPos);
	}
	if (glfwGetKey(app->window, GLFW_KEY_D) == GLFW_PRESS)
	{
		vec3 right;
		glm_cross(app->cameraFront, app->cameraUp, right);
		glm_normalize(right);
		glm_vec3_scale(right, cameraSpeed, right);
		glm_vec3_add(app->cameraPos, right, app->cameraPos);
	}
	if (glfwGetKey(app->window, GLFW_KEY_Q) == GLFW_PRESS)
	{
		vec3 up;
		glm_vec3_scale(app->cameraUp, cameraSpeed, up);
		glm_vec3_add(app->cameraPos, up, app->cameraPos);
	}
	if (glfwGetKey(app->window, GLFW_KEY_E) == GLFW_PRESS)
	{
		vec3 up;
		glm_vec3_scale(app->cameraUp, cameraSpeed, up);
		glm_vec3_sub(app->cameraPos, up, app->cameraPos);
	}
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	Application* app = glfwGetWindowUserPointer(window);

	if (app->firstMouse)
	{
		app->lastX = xpos;
		app->lastY = ypos;
		app->firstMouse = false;
	}

	float xoffset = xpos - app->lastX;
	float yoffset = app->lastY - ypos; // reversed since y-coordinates go from bottom to top
	app->lastX = xpos;
	app->lastY = ypos;

	float sensitivity = 0.1f;
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	app->yaw += xoffset;
	app->pitch += yoffset;

	if (app->pitch > 89.0f)
		app->pitch = 89.0f;
	if (app->pitch < -89.0f)
		app->pitch = -89.0f;

	vec3 front;
	front[0] = cos(glm_rad(app->yaw)) * cos(glm_rad(app->pitch));
	front[1] = sin(glm_rad(app->pitch));
	front[2] = sin(glm_rad(app->yaw)) * cos(glm_rad(app->pitch));
	glm_normalize(front);
	glm_vec3_copy(front, app->cameraFront);
}

void updateLights(Application* app)
{
	float time = glfwGetTime();

	// Animate light positions in a circular pattern
	float radius = 3.0f;

	// Light 1 - Red, circular motion in XZ plane
	app->lights[0].position[0] = cosf(time) * radius;
	app->lights[0].position[2] = sinf(time) * radius;

	// Light 2 - Green, circular motion in XZ plane (offset)
	app->lights[1].position[0] = cosf(time + 3.14159265f) * radius;
	app->lights[1].position[2] = sinf(time + 3.14159265f) * radius;

	// Light 3 - Blue, up and down motion
	app->lights[2].position[1] = 1.0f + sinf(time * 2.0f) * 1.5f;

	// Light 4 - White, diagonal circular motion
	app->lights[3].position[0] = cosf(time * 0.5f) * radius * 0.7f;
	app->lights[3].position[1] = 1.0f + sinf(time * 0.7f) * 1.0f;
	app->lights[3].position[2] = sinf(time * 0.5f) * radius * 0.7f;
}
