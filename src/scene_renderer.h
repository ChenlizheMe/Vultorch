#pragma once

/// SceneRenderer — offscreen 3D renderer for textured plane with MSAA,
///                 Blinn-Phong lighting, and orbit camera.
///
/// Renders to an offscreen framebuffer, resolves MSAA, and exposes the
/// result as an ImGui texture for embedding in UI panels.

#include <vulkan/vulkan.h>
#include "math_types.h"
#include "tensor_texture.h"
#include <cstdint>
#include <vector>

namespace vultorch {

// ── Camera ────────────────────────────────────────────────────────────
struct OrbitCamera {
    float azimuth    = 0.0f;      // horizontal angle (rad)
    float elevation  = 0.6f;      // vertical angle (rad)
    float distance   = 3.0f;
    vec3  target     = {0, 0, 0};
    float fov        = 45.0f;     // degrees
    float near_clip  = 0.01f;
    float far_clip   = 100.0f;

    vec3 eye_position() const {
        float ce = std::cos(elevation), se = std::sin(elevation);
        float ca = std::cos(azimuth),   sa = std::sin(azimuth);
        return target + vec3{ce * sa, se, ce * ca} * distance;
    }

    mat4 view_matrix() const {
        return mat4::look_at(eye_position(), target, {0, 1, 0});
    }

    mat4 proj_matrix(float aspect) const {
        return mat4::perspective(radians(fov), aspect, near_clip, far_clip);
    }

    void reset() {
        azimuth = 0; elevation = 0.6f; distance = 3.0f;
        target = {0, 0, 0};
    }
};

// ── Light ─────────────────────────────────────────────────────────────
struct LightParams {
    vec3  direction = {0.3f, -1.0f, 0.5f};
    float intensity = 1.0f;
    vec3  color     = {1.0f, 1.0f, 1.0f};
    float ambient   = 0.15f;
    float specular  = 0.5f;
    float shininess = 32.0f;
    bool  enabled   = true;
};

// ── Plane config ──────────────────────────────────────────────────────
struct PlaneConfig {
    float width  = 2.0f;    // world-space width
    float height = 2.0f;    // world-space height
    int   subdivisions = 1; // grid subdivision
};

// ── GPU UBO layouts (must match GLSL) ─────────────────────────────────
struct alignas(16) SceneUBO {
    float model[16];
    float view[16];
    float proj[16];
    float cameraPos[3];
    float _pad0;
};

struct alignas(16) LightUBO {
    float direction[3];
    float intensity;
    float color[3];
    float ambient;
    float specular;
    float shininess;
    float _pad0;
    float _pad1;
};

// ===================================================================
class SceneRenderer {
public:
    SceneRenderer() = default;
    ~SceneRenderer();

    void init(VkInstance instance, VkPhysicalDevice physDevice,
              VkDevice device, VkQueue queue, uint32_t queueFamily,
              VkDescriptorPool imguiPool,
              uint32_t width, uint32_t height,
              VkSampleCountFlagBits msaa = VK_SAMPLE_COUNT_4_BIT);

    void resize(uint32_t width, uint32_t height);
    void set_msaa(VkSampleCountFlagBits msaa);

    /// Prepare rendering state (UBOs, descriptors, pipeline).
    /// Called during ImGui frame; actual GPU recording is deferred.
    void prepare(TensorTexture& texture);

    /// Record offscreen render commands into an external command buffer.
    /// Called by Engine::end_frame() — single submit, zero stalls.
    void record_render(VkCommandBuffer cmd);

    /// Whether this renderer needs to record commands this frame.
    bool needs_render() const { return needs_render_; }

    /// ImGui texture ID of the resolved result.
    uintptr_t imgui_texture_id() const;

    /// Offscreen dimensions.
    uint32_t width()  const { return width_; }
    uint32_t height() const { return height_; }

    /// Handle mouse interaction for orbit camera.
    /// Call each frame before render().
    void process_input(float mouse_dx, float mouse_dy, float scroll,
                       bool left_btn, bool right_btn, bool middle_btn);

    // Public parameters
    OrbitCamera camera;
    LightParams light;
    PlaneConfig plane;
    float background[3] = {0.12f, 0.12f, 0.14f};

    void destroy();

private:
    // Vulkan handles (borrowed)
    VkInstance       instance_      = VK_NULL_HANDLE;
    VkPhysicalDevice phys_device_   = VK_NULL_HANDLE;
    VkDevice         device_        = VK_NULL_HANDLE;
    VkQueue          queue_         = VK_NULL_HANDLE;
    uint32_t         queue_family_  = 0;
    VkDescriptorPool imgui_pool_    = VK_NULL_HANDLE;

    // Offscreen dimensions
    uint32_t width_  = 0;
    uint32_t height_ = 0;
    VkSampleCountFlagBits msaa_samples_ = VK_SAMPLE_COUNT_4_BIT;

    // ── Offscreen framebuffer ──────────────────────────────────────
    // MSAA color attachment
    VkImage        msaa_color_        = VK_NULL_HANDLE;
    VkDeviceMemory msaa_color_mem_    = VK_NULL_HANDLE;
    VkImageView    msaa_color_view_   = VK_NULL_HANDLE;

    // MSAA depth attachment
    VkImage        msaa_depth_        = VK_NULL_HANDLE;
    VkDeviceMemory msaa_depth_mem_    = VK_NULL_HANDLE;
    VkImageView    msaa_depth_view_   = VK_NULL_HANDLE;

    // Resolve target (1x) — sampled by ImGui
    VkImage        resolve_color_     = VK_NULL_HANDLE;
    VkDeviceMemory resolve_color_mem_ = VK_NULL_HANDLE;
    VkImageView    resolve_view_      = VK_NULL_HANDLE;

    VkRenderPass  render_pass_  = VK_NULL_HANDLE;
    VkFramebuffer framebuffer_  = VK_NULL_HANDLE;

    // ImGui texture for resolved result
    VkSampler       resolve_sampler_ = VK_NULL_HANDLE;
    VkDescriptorSet imgui_desc_      = VK_NULL_HANDLE;

    // ── Pipeline ───────────────────────────────────────────────────
    VkPipelineLayout      pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline            pipeline_        = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout_     = VK_NULL_HANDLE;
    VkDescriptorPool      scene_pool_      = VK_NULL_HANDLE;
    VkDescriptorSet       scene_desc_      = VK_NULL_HANDLE;

    // ── Plane mesh ─────────────────────────────────────────────────
    VkBuffer       vertex_buf_ = VK_NULL_HANDLE;
    VkDeviceMemory vertex_mem_ = VK_NULL_HANDLE;
    VkBuffer       index_buf_  = VK_NULL_HANDLE;
    VkDeviceMemory index_mem_  = VK_NULL_HANDLE;
    uint32_t       index_count_ = 0;

    // Deferred render flag
    bool needs_render_ = false;

    // ── UBOs ───────────────────────────────────────────────────────
    VkBuffer       ubo_buf_    = VK_NULL_HANDLE;
    VkDeviceMemory ubo_mem_    = VK_NULL_HANDLE;
    void*          ubo_mapped_ = nullptr;

    VkBuffer       light_ubo_buf_    = VK_NULL_HANDLE;
    VkDeviceMemory light_ubo_mem_    = VK_NULL_HANDLE;
    void*          light_ubo_mapped_ = nullptr;

    bool initialized_ = false;

    // ── Internal helpers ───────────────────────────────────────────
    void create_offscreen_resources();
    void create_render_pass();
    void create_framebuffer();
    void create_pipeline();
    void create_plane_mesh();
    void create_ubos();
    void create_descriptor_set(TensorTexture& texture);
    void update_ubos();
    void cleanup_offscreen();

    VkShaderModule create_shader_module(const uint32_t* code, size_t size);
    uint32_t find_memory_type(uint32_t filter, VkMemoryPropertyFlags props);

    void create_image(uint32_t w, uint32_t h, VkFormat format,
                      VkSampleCountFlagBits samples, VkImageUsageFlags usage,
                      VkImage& image, VkDeviceMemory& memory);
    void create_image_view(VkImage image, VkFormat format,
                           VkImageAspectFlags aspect, VkImageView& view);
};

} // namespace vultorch
