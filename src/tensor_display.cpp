#ifdef VULTORCH_HAS_CUDA

#include "tensor_display.h"
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <cuda.h>
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>

namespace vultorch {

#define VK_CHECK(x)                                                          \
    do {                                                                     \
        VkResult _r = (x);                                                   \
        if (_r != VK_SUCCESS)                                                \
            throw std::runtime_error(                                        \
                std::string("Vulkan error ") + std::to_string((int)_r) +     \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));         \
    } while (0)

#define CU_CHECK(x)                                                          \
    do {                                                                     \
        CUresult _r = (x);                                                   \
        if (_r != CUDA_SUCCESS) {                                            \
            const char* msg = nullptr;                                       \
            cuGetErrorString(_r, &msg);                                      \
            throw std::runtime_error(                                        \
                std::string("CUDA error: ") + (msg ? msg : "unknown") +      \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));         \
        }                                                                    \
    } while (0)

// ======================================================================
TensorDisplay::~TensorDisplay() {
    destroy();
}

void TensorDisplay::init(VkInstance instance, VkPhysicalDevice physDevice,
                         VkDevice device, VkQueue queue, uint32_t queueFamily,
                         VkRenderPass /*renderPass*/, VkDescriptorPool imguiPool) {
    instance_     = instance;
    phys_device_  = physDevice;
    device_       = device;
    queue_        = queue;
    queue_family_ = queueFamily;
    imgui_pool_   = imguiPool;

    // Command pool for one-shot commands
    VkCommandPoolCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = queueFamily;
    VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &cmd_pool_));

    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = cmd_pool_;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(device_, &ai, &cmd_buf_));

    // Sampler (nearest for pixel-accurate display, linear also fine)
    VkSamplerCreateInfo si{};
    si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter    = VK_FILTER_LINEAR;
    si.minFilter    = VK_FILTER_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device_, &si, nullptr, &sampler_));

    // Init CUDA driver API
    CU_CHECK(cuInit(0));
}

void TensorDisplay::destroy() {
    if (!device_) return;
    vkDeviceWaitIdle(device_);
    free_resources();
    if (sampler_)  { vkDestroySampler(device_, sampler_, nullptr);       sampler_  = VK_NULL_HANDLE; }
    if (cmd_pool_) { vkDestroyCommandPool(device_, cmd_pool_, nullptr);  cmd_pool_ = VK_NULL_HANDLE; }
    device_ = VK_NULL_HANDLE;
}

// ======================================================================
//  Upload tensor data → Vulkan image via staging buffer
// ======================================================================
void TensorDisplay::upload(uintptr_t data_ptr, uint32_t width, uint32_t height,
                           uint32_t channels, int device_id) {
    if (width == 0 || height == 0) return;
    if (channels != 3 && channels != 4)
        throw std::runtime_error("TensorDisplay: channels must be 3 or 4");

    // Re-allocate if size changed
    if (!allocated_ || width_ != width || height_ != height || channels_ != channels) {
        free_resources();
        allocate_resources(width, height, channels);
    }

    // ── Copy from CUDA device memory → staging buffer ──────────────
    CUdevice cu_dev;
    CU_CHECK(cuDeviceGet(&cu_dev, device_id));

    // We need a CUDA context. Try to use the current one, or create one.
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    if (!ctx) {
        CU_CHECK(cuCtxCreate(&ctx, 0, cu_dev));
    }

    // Tensor layout: [H, W, C] contiguous float32
    // If 3-channel, we need to expand to RGBA for Vulkan
    size_t src_bytes = (size_t)width * height * channels * sizeof(float);

    if (channels == 4) {
        // Direct copy: CUDA device → host staging
        CU_CHECK(cuMemcpyDtoH(staging_mapped_, (CUdeviceptr)data_ptr, src_bytes));
    } else {
        // 3-channel: copy to temp host buffer, then expand to RGBA
        std::vector<float> tmp(width * height * 3);
        CU_CHECK(cuMemcpyDtoH(tmp.data(), (CUdeviceptr)data_ptr, src_bytes));

        float* dst = static_cast<float*>(staging_mapped_);
        for (uint32_t i = 0; i < width * height; i++) {
            dst[i * 4 + 0] = tmp[i * 3 + 0];
            dst[i * 4 + 1] = tmp[i * 3 + 1];
            dst[i * 4 + 2] = tmp[i * 3 + 2];
            dst[i * 4 + 3] = 1.0f;
        }
    }

    // ── Upload staging → VkImage ───────────────────────────────────
    transition_image_layout(image_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copy_buffer_to_image(staging_buf_, image_, width, height);
    transition_image_layout(image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

uintptr_t TensorDisplay::imgui_texture_id() const {
    return reinterpret_cast<uintptr_t>(descriptor_set_);
}

// ======================================================================
//  Internal: allocate Vulkan resources
// ======================================================================
void TensorDisplay::allocate_resources(uint32_t w, uint32_t h, uint32_t ch) {
    width_    = w;
    height_   = h;
    channels_ = ch;

    // Always use RGBA float32 for the Vulkan image
    VkFormat format       = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkDeviceSize img_size = (VkDeviceSize)w * h * 4 * sizeof(float);

    // ── VkImage ──────────────────────────────────────────────────
    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = format;
    ici.extent        = {w, h, 1};
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(device_, &ici, nullptr, &image_));

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device_, image_, &memReq);

    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = memReq.size;
    mai.memoryTypeIndex = find_memory_type(memReq.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device_, &mai, nullptr, &memory_));
    VK_CHECK(vkBindImageMemory(device_, image_, memory_, 0));

    // ── VkImageView ──────────────────────────────────────────────
    VkImageViewCreateInfo vci{};
    vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image    = image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = format;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device_, &vci, nullptr, &view_));

    // ── Staging buffer (host-visible) ────────────────────────────
    staging_size_ = img_size;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = img_size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VK_CHECK(vkCreateBuffer(device_, &bci, nullptr, &staging_buf_));

    VkMemoryRequirements bufReq;
    vkGetBufferMemoryRequirements(device_, staging_buf_, &bufReq);

    VkMemoryAllocateInfo bmai{};
    bmai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bmai.allocationSize  = bufReq.size;
    bmai.memoryTypeIndex = find_memory_type(bufReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(device_, &bmai, nullptr, &staging_mem_));
    VK_CHECK(vkBindBufferMemory(device_, staging_buf_, staging_mem_, 0));
    VK_CHECK(vkMapMemory(device_, staging_mem_, 0, img_size, 0, &staging_mapped_));

    // ── ImGui descriptor set (texture ID) ────────────────────────
    descriptor_set_ = ImGui_ImplVulkan_AddTexture(sampler_, view_,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    allocated_ = true;
}

void TensorDisplay::free_resources() {
    if (!allocated_) return;
    vkDeviceWaitIdle(device_);

    if (descriptor_set_) {
        ImGui_ImplVulkan_RemoveTexture(descriptor_set_);
        descriptor_set_ = VK_NULL_HANDLE;
    }
    if (staging_mapped_) {
        vkUnmapMemory(device_, staging_mem_);
        staging_mapped_ = nullptr;
    }
    if (staging_buf_) { vkDestroyBuffer(device_, staging_buf_, nullptr);  staging_buf_ = VK_NULL_HANDLE; }
    if (staging_mem_) { vkFreeMemory(device_, staging_mem_, nullptr);     staging_mem_ = VK_NULL_HANDLE; }
    if (view_)        { vkDestroyImageView(device_, view_, nullptr);      view_        = VK_NULL_HANDLE; }
    if (image_)       { vkDestroyImage(device_, image_, nullptr);         image_       = VK_NULL_HANDLE; }
    if (memory_)      { vkFreeMemory(device_, memory_, nullptr);          memory_      = VK_NULL_HANDLE; }

    allocated_ = false;
}

// ======================================================================
//  Helpers
// ======================================================================
uint32_t TensorDisplay::find_memory_type(uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(phys_device_, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
        if ((filter & (1u << i)) && (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

void TensorDisplay::transition_image_layout(VkImage image, VkImageLayout oldL, VkImageLayout newL) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkResetCommandBuffer(cmd_buf_, 0));
    VK_CHECK(vkBeginCommandBuffer(cmd_buf_, &bi));

    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = oldL;
    barrier.newLayout           = newL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags srcStage, dstStage;

    if (oldL == VK_IMAGE_LAYOUT_UNDEFINED && newL == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldL == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newL == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        throw std::runtime_error("Unsupported layout transition");
    }

    vkCmdPipelineBarrier(cmd_buf_, srcStage, dstStage, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);

    VK_CHECK(vkEndCommandBuffer(cmd_buf_));

    VkSubmitInfo submit{};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd_buf_;
    VK_CHECK(vkQueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue_));
}

void TensorDisplay::copy_buffer_to_image(VkBuffer buf, VkImage img, uint32_t w, uint32_t h) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkResetCommandBuffer(cmd_buf_, 0));
    VK_CHECK(vkBeginCommandBuffer(cmd_buf_, &bi));

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {w, h, 1};

    vkCmdCopyBufferToImage(cmd_buf_, buf, img,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    VK_CHECK(vkEndCommandBuffer(cmd_buf_));

    VkSubmitInfo submit{};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd_buf_;
    VK_CHECK(vkQueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue_));
}

} // namespace vultorch

#endif // VULTORCH_HAS_CUDA
