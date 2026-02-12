#pragma once

// SDL_MAIN_HANDLED is defined via CMake
// #define SDL_MAIN_HANDLED
#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#include <vulkan/vulkan.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <memory>

#ifdef VULTORCH_HAS_CUDA
#include "tensor_texture.h"
#include "scene_renderer.h"
#endif

namespace vultorch {

class Engine {
public:
    Engine() = default;
    ~Engine();

    void init(const char* title, int width, int height);
    void destroy();

    /// Process window events. Returns false when the window should close.
    bool poll();

    /// Begin a new frame. Returns false if the frame was skipped (e.g. minimized / swapchain recreate).
    bool begin_frame();

    /// End the current frame and present.
    void end_frame();

#ifdef VULTORCH_HAS_CUDA
    /// Get or lazily create the tensor texture.
    TensorTexture& tensor_texture();

    /// Get or lazily create the scene renderer.
    SceneRenderer& scene_renderer(uint32_t width = 800, uint32_t height = 600,
                                  VkSampleCountFlagBits msaa = VK_SAMPLE_COUNT_4_BIT);

    /// Expose the current command buffer (valid between begin_frame/end_frame).
    VkCommandBuffer current_cmd() const { return command_buffers_[frame_index_]; }

    // Expose Vulkan handles for sub-systems
    VkInstance       vk_instance()       const { return instance_; }
    VkPhysicalDevice vk_physical_device() const { return physical_device_; }
    VkDevice         vk_device()          const { return device_; }
    VkQueue          vk_queue()           const { return graphics_queue_; }
    uint32_t         vk_queue_family()    const { return graphics_family_; }
    VkDescriptorPool vk_imgui_pool()      const { return imgui_pool_; }
    VkRenderPass     vk_render_pass()     const { return render_pass_; }

    /// Query max supported MSAA sample count.
    VkSampleCountFlagBits max_msaa_samples() const;
#endif

private:
    // ---- SDL ----
    SDL_Window* window_ = nullptr;

    // ---- Vulkan core ----
    VkInstance       instance_        = VK_NULL_HANDLE;
    VkSurfaceKHR     surface_         = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice         device_          = VK_NULL_HANDLE;
    VkQueue          graphics_queue_  = VK_NULL_HANDLE;
    uint32_t         graphics_family_ = 0;

    // ---- Swapchain ----
    VkSwapchainKHR              swapchain_        = VK_NULL_HANDLE;
    VkFormat                    swapchain_format_  = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D                  swapchain_extent_  = {};
    std::vector<VkImage>        swapchain_images_;
    std::vector<VkImageView>    swapchain_views_;
    std::vector<VkFramebuffer>  framebuffers_;

    // ---- Render pass ----
    VkRenderPass render_pass_ = VK_NULL_HANDLE;

    // ---- Commands ----
    VkCommandPool                command_pool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers_;

    // ---- Sync ----
    static constexpr uint32_t MAX_FRAMES = 2;
    std::vector<VkSemaphore> sem_available_;
    std::vector<VkSemaphore> sem_finished_;
    std::vector<VkFence>     fences_;
    uint32_t frame_index_ = 0;
    uint32_t image_index_ = 0;

    // ---- ImGui ----
    VkDescriptorPool imgui_pool_ = VK_NULL_HANDLE;

#ifdef VULTORCH_HAS_CUDA
    // ---- Tensor texture (zero-copy) ----
    std::unique_ptr<TensorTexture> tensor_texture_;
    // ---- 3D Scene renderer ----
    std::unique_ptr<SceneRenderer> scene_renderer_;
#endif

    // ---- State flags ----
    bool initialized_  = false;
    bool frame_started_ = false;

    // ---- Internal helpers ----
    void create_instance();
    void create_surface();
    void pick_physical_device();
    void create_device();
    void create_swapchain();
    void create_render_pass();
    void create_framebuffers();
    void create_commands();
    void create_sync();
    void init_imgui();
    void cleanup_swapchain();
    void recreate_swapchain();
};

} // namespace vultorch
