#ifndef GRAPHICS_H
#define GRAPHICS_H

#include "vulkan_core.h"
#include "external/cglm/include/cglm/cglm.h"

// Vertex structure
typedef struct Vertex {
    vec3 pos;
    vec3 normal;
    vec2 texcoord;
} Vertex;

// Mesh structure
typedef struct Mesh {
    Vertex* vertices;
    u32* indices;
    u32 vertexCount;
    u32 indexCount;
    Buffer vertexBuffer;
    Buffer indexBuffer;
} Mesh;

// Camera structure
typedef struct Camera {
    vec3 position;
    vec3 front;
    vec3 up;
    float fov;
    float speed;
    float sensitivity;
    float yaw;
    float pitch;
    bool firstMouse;
    double lastX, lastY;
} Camera;

// Scene structure
typedef struct Scene {
    Mesh* meshes;
    u32 meshCount;
    Texture* textures;
    u32 textureCount;
    Camera camera;
    Buffer uniformBuffer;
    mat4 model;
    mat4 view;
    mat4 projection;
} Scene;

// Graphics functions
bool graphics_init(Scene* scene, VulkanContext* ctx);
void graphics_cleanup(Scene* scene, VulkanContext* ctx);
bool graphics_load_model(Scene* scene, VulkanContext* ctx, const char* path);
void graphics_update_camera(Camera* camera, GLFWwindow* window, float deltaTime);
void graphics_render_frame(Scene* scene, VulkanContext* ctx, float deltaTime);
void graphics_update_uniforms(Scene* scene, u32 width, u32 height);

// Camera functions
void camera_init(Camera* camera);
void camera_process_keyboard(Camera* camera, GLFWwindow* window, float deltaTime);
void camera_process_mouse(Camera* camera, double xpos, double ypos);
void camera_process_scroll(Camera* camera, double yoffset);

#endif // GRAPHICS_H