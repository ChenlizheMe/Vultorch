#pragma once

#ifdef VULTORCH_HAS_CUDA

#include <vulkan/vulkan.h>
#include <cuda.h>
#include <cstdint>

namespace vultorch {

/// Wraps a Vulkan image + CUDA external memory import for zero-copy tensor display.
/// The user provides a CUDA device pointer (from torch.Tensor.data_ptr()),
/// and this class maps it into a VkImage that ImGui can sample as a texture.
class TensorDisplay {
public:
    TensorDisplay() = default;
    ~TensorDisplay();

    /// Initialise with existing Vulkan handles.
    void init(VkInstance instance, VkPhysicalDevice physDevice,
              VkDevice device, VkQueue queue, uint32_t queueFamily,
              VkRenderPass renderPass, VkDescriptorPool imguiPool);

    /// Upload a CUDA tensor to the Vulkan texture.
    /// @param data_ptr   GPU pointer from tensor.data_ptr()
    /// @param width      image width in pixels
    /// @param height     image height in pixels
    /// @param channels   3 (RGB) or 4 (RGBA), float32 per channel
    /// @param device_id  CUDA device ordinal (usually 0)
    void upload(uintptr_t data_ptr, uint32_t width, uint32_t height,
                uint32_t channels, int device_id = 0);

    /// Returns the ImGui texture ID for use with ui.image().
    /// Returns 0 (nullptr) if no texture has been uploaded yet.
    uintptr_t imgui_texture_id() const;

    /// Current texture dimensions.
    uint32_t width()  const { return width_; }
    uint32_t height() const { return height_; }

    void destroy();

private:
    // Vulkan handles (borrowed, not owned)
    VkInstance       instance_       = VK_NULL_HANDLE;
    VkPhysicalDevice phys_device_    = VK_NULL_HANDLE;
    VkDevice         device_         = VK_NULL_HANDLE;
    VkQueue          queue_          = VK_NULL_HANDLE;
    uint32_t         queue_family_   = 0;
    VkDescriptorPool imgui_pool_     = VK_NULL_HANDLE;

    // Vulkan texture resources (owned)
    VkImage          image_          = VK_NULL_HANDLE;
    VkDeviceMemory   memory_         = VK_NULL_HANDLE;
    VkImageView      view_           = VK_NULL_HANDLE;
    VkSampler        sampler_        = VK_NULL_HANDLE;
    VkDescriptorSet  descriptor_set_ = VK_NULL_HANDLE;

    // Staging buffer for CPU-path fallback and CUDA memcpy
    VkBuffer         staging_buf_    = VK_NULL_HANDLE;
    VkDeviceMemory   staging_mem_    = VK_NULL_HANDLE;
    void*            staging_mapped_ = nullptr;
    VkDeviceSize     staging_size_   = 0;

    // Command buffer for layout transitions + copies
    VkCommandPool    cmd_pool_       = VK_NULL_HANDLE;
    VkCommandBuffer  cmd_buf_        = VK_NULL_HANDLE;

    // Dimensions
    uint32_t width_     = 0;
    uint32_t height_    = 0;
    uint32_t channels_  = 4;
    bool     allocated_ = false;

    void allocate_resources(uint32_t w, uint32_t h, uint32_t ch);
    void free_resources();

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props);
    void transition_image_layout(VkImage image, VkImageLayout oldL, VkImageLayout newL);
    void copy_buffer_to_image(VkBuffer buf, VkImage img, uint32_t w, uint32_t h);
};

} // namespace vultorch

#endif // VULTORCH_HAS_CUDA
