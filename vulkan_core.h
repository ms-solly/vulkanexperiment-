#ifndef VULKAN_CORE_H
#define VULKAN_CORE_H

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <stdint.h>
#include <stdbool.h>

#define u32 uint32_t

// Forward declarations
typedef struct VulkanContext VulkanContext;
typedef struct Buffer Buffer;
typedef struct Texture Texture;

// Buffer structure
typedef struct Buffer {
    VkBuffer vkbuffer;
    VkDeviceMemory memory;
    void* data;
    size_t size;
} Buffer;

// Texture structure
typedef struct Texture {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
    VkSampler sampler;
} Texture;

// Main Vulkan context
typedef struct VulkanContext {
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue graphicsQueue;
    u32 queueFamilyIndex;
    VkCommandPool commandPool;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    
    // Swapchain resources
    VkImage* swapchainImages;
    VkImageView* swapchainImageViews;
    VkFramebuffer* framebuffers;
    u32 swapchainImageCount;
    VkFormat swapchainImageFormat;
    VkExtent2D swapchainExtent;
    
    // Depth resources
    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    VkFormat depthFormat;
    
    // Synchronization
    VkSemaphore imageAvailableSemaphore;
    VkSemaphore renderFinishedSemaphore;
    
    // Memory properties
    VkPhysicalDeviceMemoryProperties memoryProperties;
} VulkanContext;

// Core Vulkan functions
bool vulkan_init(VulkanContext* ctx, GLFWwindow* window, u32 width, u32 height);
void vulkan_cleanup(VulkanContext* ctx);

// Resource management
bool vulkan_create_buffer(VulkanContext* ctx, Buffer* buffer, size_t size, VkBufferUsageFlags usage);
void vulkan_destroy_buffer(VulkanContext* ctx, Buffer* buffer);
bool vulkan_create_texture(VulkanContext* ctx, Texture* texture, const char* path);
void vulkan_destroy_texture(VulkanContext* ctx, Texture* texture);

// Rendering
bool vulkan_begin_frame(VulkanContext* ctx, u32* imageIndex);
bool vulkan_end_frame(VulkanContext* ctx, u32 imageIndex);
VkCommandBuffer vulkan_begin_command_buffer(VulkanContext* ctx);
void vulkan_end_command_buffer(VulkanContext* ctx, VkCommandBuffer commandBuffer);

// Pipeline management
bool vulkan_create_graphics_pipeline(VulkanContext* ctx, const char* vertShaderPath, const char* fragShaderPath);
void vulkan_update_descriptor_set(VulkanContext* ctx, Buffer* uniformBuffer, Texture* texture);

#endif // VULKAN_CORE_H