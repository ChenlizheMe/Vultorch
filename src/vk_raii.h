#pragma once
/// vk_raii.h — Lightweight RAII wrapper for Vulkan handles.
///
/// Usage:
///   VkUniqueBuffer buf(device, raw_buf);        // takes ownership
///   VkBuffer raw = buf.get();                     // access underlying handle
///   buf.reset();                                  // explicit destroy + null
///   // auto-destroyed on scope exit
///
/// The wrapper only calls the deleter if the handle is non-null.
/// Move-only semantics (no copies).

#include <vulkan/vulkan.h>
#include <utility>

namespace vultorch {

/// Generic RAII wrapper for a Vulkan handle destroyed via vkDestroyXxx(device, handle, nullptr).
template<typename T, void (*Deleter)(VkDevice, T, const VkAllocationCallbacks*)>
class VkHandle {
public:
    VkHandle() = default;
    VkHandle(VkDevice device, T handle) : device_(device), handle_(handle) {}
    ~VkHandle() { reset(); }

    // Move only
    VkHandle(VkHandle&& o) noexcept : device_(o.device_), handle_(o.handle_) {
        o.handle_ = VK_NULL_HANDLE;
    }
    VkHandle& operator=(VkHandle&& o) noexcept {
        if (this != &o) {
            reset();
            device_ = o.device_;
            handle_ = o.handle_;
            o.handle_ = VK_NULL_HANDLE;
        }
        return *this;
    }
    VkHandle(const VkHandle&) = delete;
    VkHandle& operator=(const VkHandle&) = delete;

    void reset() {
        if (handle_ != VK_NULL_HANDLE) {
            Deleter(device_, handle_, nullptr);
            handle_ = VK_NULL_HANDLE;
        }
    }

    /// Release ownership without destroying.
    T release() {
        T tmp = handle_;
        handle_ = VK_NULL_HANDLE;
        return tmp;
    }

    /// Assign a new handle (destroys any previous one).
    void reset(VkDevice device, T handle) {
        reset();
        device_ = device;
        handle_ = handle;
    }

    T  get()  const { return handle_; }
    T* ptr()        { return &handle_; }
    operator T()    const { return handle_; }
    explicit operator bool() const { return handle_ != VK_NULL_HANDLE; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    T handle_        = VK_NULL_HANDLE;
};

// ── Convenience aliases ───────────────────────────────────────────────
using VkUniqueBuffer       = VkHandle<VkBuffer,              vkDestroyBuffer>;
using VkUniqueImage        = VkHandle<VkImage,               vkDestroyImage>;
using VkUniqueImageView    = VkHandle<VkImageView,           vkDestroyImageView>;
using VkUniqueSampler      = VkHandle<VkSampler,             vkDestroySampler>;
using VkUniquePipeline     = VkHandle<VkPipeline,            vkDestroyPipeline>;
using VkUniquePipelineLayout = VkHandle<VkPipelineLayout,    vkDestroyPipelineLayout>;
using VkUniqueDescriptorSetLayout = VkHandle<VkDescriptorSetLayout, vkDestroyDescriptorSetLayout>;
using VkUniqueDescriptorPool = VkHandle<VkDescriptorPool,    vkDestroyDescriptorPool>;
using VkUniqueRenderPass   = VkHandle<VkRenderPass,          vkDestroyRenderPass>;
using VkUniqueFramebuffer  = VkHandle<VkFramebuffer,         vkDestroyFramebuffer>;
using VkUniqueFence        = VkHandle<VkFence,               vkDestroyFence>;
using VkUniqueSemaphore    = VkHandle<VkSemaphore,           vkDestroySemaphore>;
using VkUniqueCommandPool  = VkHandle<VkCommandPool,         vkDestroyCommandPool>;
using VkUniqueShaderModule = VkHandle<VkShaderModule,        vkDestroyShaderModule>;
using VkUniqueSwapchain    = VkHandle<VkSwapchainKHR,        vkDestroySwapchainKHR>;

/// RAII wrapper for VkDeviceMemory (uses vkFreeMemory instead of vkDestroyXxx).
using VkUniqueDeviceMemory = VkHandle<VkDeviceMemory, vkFreeMemory>;

} // namespace vultorch
