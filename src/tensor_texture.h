#pragma once

/// TensorTexture — Vulkan texture for tensor display.
///
/// Three usage modes:
///   1. allocate_shared() [CUDA only] → returns a CUDA device pointer.
///      Wrap with torch.as_tensor() on Python side – writes go directly to
///      Vulkan-visible memory.  Call sync() each frame.
///
///   2. upload() [CUDA only] → GPU-GPU copy from any CUDA tensor into the
///      shared buffer, then buffer → image.
///
///   3. upload_cpu() [always available] → CPU memcpy into host-visible
///      staging buffer, then buffer → image.
///
/// Modes 1 & 2 avoid any GPU → CPU transfer.
/// Mode 3 supports CPU-only PyTorch installations.

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <vulkan/vulkan.h>
#include <cstdint>

#ifdef VULTORCH_HAS_CUDA
#include <cuda.h>
#endif

namespace vultorch {

enum class FilterMode { Nearest, Linear };

class TensorTexture {
public:
    TensorTexture() = default;
    ~TensorTexture();

    /// Initialise with existing Vulkan handles.
    void init(VkInstance instance, VkPhysicalDevice physDevice,
              VkDevice device, VkQueue queue, uint32_t queueFamily,
              VkDescriptorPool imguiPool);

#ifdef VULTORCH_HAS_CUDA
    /// Allocate a Vulkan buffer with external-memory export, import into CUDA,
    /// and return the CUDA device pointer.  The caller can wrap this with
    /// torch.as_tensor() for zero-copy writes.
    /// Returns 0 on failure.
    uintptr_t allocate_shared(uint32_t width, uint32_t height,
                              uint32_t channels, int cuda_device = 0);

    /// Upload from an arbitrary CUDA device pointer (GPU-GPU memcpy).
    void upload(uintptr_t data_ptr, uint32_t width, uint32_t height,
                uint32_t channels, int cuda_device = 0);

    /// Sync shared buffer → VkImage.  Call once per frame when using the
    /// allocate_shared() path.  Only 4-channel (RGBA) is supported.
    void sync();

    /// Check whether the given data_ptr is the shared pointer
    /// (i.e. the tensor was created via allocate_shared).
    bool is_shared_ptr(uintptr_t data_ptr) const {
        return shared_cuda_ptr_ != 0 && data_ptr == shared_cuda_ptr_;
    }
#endif

    /// Upload from CPU memory (host memcpy → staging buffer → VkImage).
    /// The data must be tightly-packed RGBA float32 (channels must be 4).
    /// Python show() handles RGBA expansion before calling this.
    void upload_cpu(const void* data, uint32_t width, uint32_t height,
                    uint32_t channels);

    /// Record copy commands into an external command buffer (deferred path).
    /// Used by Engine::end_frame() to integrate the copy into the main submit,
    /// avoiding a separate vkQueueSubmit.
    void record_copy(VkCommandBuffer cmd);

    /// Standalone submit + fence-wait: copy staging→image using own command
    /// buffer.  For SceneRenderer or any path that needs the image ready
    /// before end_frame().
    void flush();

    /// Wait for any pending standalone copy fence.
    void wait_for_copy();

    /// Whether a staging→image copy is pending.
    bool needs_copy() const { return needs_copy_; }

    /// ImGui texture ID.
    VkDescriptorSet imgui_descriptor() const { return descriptor_set_; }
    uintptr_t       imgui_texture_id() const { return reinterpret_cast<uintptr_t>(descriptor_set_); }

    /// Image view & sampler for use in custom pipelines (e.g. SceneRenderer).
    VkImageView image_view() const { return view_; }
    VkSampler   sampler()    const { return sampler_; }

    /// Dimensions.
    uint32_t width()    const { return width_; }
    uint32_t height()   const { return height_; }
    uint32_t channels() const { return channels_; }

    /// Filter mode.
    void set_filter(FilterMode mode);

    void destroy();

private:
    // Vulkan handles (borrowed)
    VkInstance       instance_      = VK_NULL_HANDLE;
    VkPhysicalDevice phys_device_   = VK_NULL_HANDLE;
    VkDevice         device_        = VK_NULL_HANDLE;
    VkQueue          queue_         = VK_NULL_HANDLE;
    uint32_t         queue_family_  = 0;
    VkDescriptorPool imgui_pool_    = VK_NULL_HANDLE;

    // VkImage (optimal tiling, destination of buffer copies)
    VkImage          image_         = VK_NULL_HANDLE;
    VkDeviceMemory   image_mem_     = VK_NULL_HANDLE;
    VkImageView      view_          = VK_NULL_HANDLE;
    VkSampler        sampler_       = VK_NULL_HANDLE;
    VkDescriptorSet  descriptor_set_ = VK_NULL_HANDLE;

    // Staging buffer (used for BOTH CUDA external-memory AND CPU host paths).
    // Only one mode is active at a time, tracked by staging_is_host_.
    VkBuffer         staging_buf_   = VK_NULL_HANDLE;
    VkDeviceMemory   staging_mem_   = VK_NULL_HANDLE;
    VkDeviceSize     staging_size_  = 0;
    void*            staging_mapped_ = nullptr;  // non-null when host-visible staging
    bool             staging_is_host_ = false;   // true = host-visible, false = external memory

#ifdef VULTORCH_HAS_CUDA
    // CUDA side of the staging buffer (external-memory path only)
    CUexternalMemory  cuda_ext_mem_    = nullptr;
    CUdeviceptr       cuda_staging_ptr_ = 0;

    // Shared pointer (returned to Python for zero-copy)
    uintptr_t         shared_cuda_ptr_ = 0;
#endif

    // Command buffer for layout transitions & copies
    VkCommandPool    cmd_pool_      = VK_NULL_HANDLE;
    VkCommandBuffer  cmd_buf_       = VK_NULL_HANDLE;

    // Fence for standalone copy submit (replaces vkQueueWaitIdle)
    VkFence          copy_fence_    = VK_NULL_HANDLE;
    bool             fence_pending_ = false;

    // Deferred copy flag — set by upload()/sync()/upload_cpu(), cleared by record_copy()/flush()
    bool             needs_copy_    = false;

    // Dimensions
    uint32_t width_     = 0;
    uint32_t height_    = 0;
    uint32_t channels_  = 0;
    bool     allocated_ = false;

    FilterMode filter_ = FilterMode::Linear;

    // --- Internal helpers ---
    void allocate_image(uint32_t w, uint32_t h);
    void allocate_staging_host(uint32_t w, uint32_t h, uint32_t ch);
#ifdef VULTORCH_HAS_CUDA
    void allocate_staging_external(uint32_t w, uint32_t h, uint32_t ch, int cuda_device);
#endif
    void free_resources();
    void recreate_sampler();
    void record_copy_commands(VkCommandBuffer cmd);  // barrier + copy + barrier
    void copy_staging_to_image();                     // standalone submit with fence
    uint32_t find_memory_type(uint32_t filter, VkMemoryPropertyFlags props);

    // Platform-specific external memory helpers (CUDA path only)
#ifdef VULTORCH_HAS_CUDA
#ifdef _WIN32
    using ExternalHandleType = HANDLE;
    static constexpr VkExternalMemoryHandleTypeFlagBits kExternalMemHandleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    using ExternalHandleType = int;
    static constexpr VkExternalMemoryHandleTypeFlagBits kExternalMemHandleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
#endif // VULTORCH_HAS_CUDA
};

} // namespace vultorch
