// TensorTexture — Vulkan texture for tensor display (CUDA + CPU paths).
// See tensor_texture.h for the three usage modes.

#include "tensor_texture.h"
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

namespace vultorch {

// ── Error-check macros ────────────────────────────────────────────────
#define VK_CHECK(x)                                                           \
    do {                                                                      \
        VkResult _r = (x);                                                    \
        if (_r != VK_SUCCESS)                                                 \
            throw std::runtime_error(                                         \
                std::string("Vulkan error ") + std::to_string((int)_r) +      \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
    } while (0)

#ifdef VULTORCH_HAS_CUDA
#define CU_CHECK(x)                                                           \
    do {                                                                      \
        CUresult _r = (x);                                                    \
        if (_r != CUDA_SUCCESS) {                                             \
            const char* msg = nullptr;                                        \
            cuGetErrorString(_r, &msg);                                       \
            throw std::runtime_error(                                         \
                std::string("CUDA error: ") + (msg ? msg : "unknown") +       \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
        }                                                                     \
    } while (0)

static bool s_cuda_initialized = false;
static void ensure_cuda_init() {
    if (!s_cuda_initialized) {
        CU_CHECK(cuInit(0));
        s_cuda_initialized = true;
    }
}
#endif // VULTORCH_HAS_CUDA

// ======================================================================
TensorTexture::~TensorTexture() { destroy(); }

void TensorTexture::init(VkInstance instance, VkPhysicalDevice physDevice,
                         VkDevice device, VkQueue queue, uint32_t queueFamily,
                         VkDescriptorPool imguiPool) {
    instance_     = instance;
    phys_device_  = physDevice;
    device_       = device;
    queue_        = queue;
    queue_family_ = queueFamily;
    imgui_pool_   = imguiPool;

    // Command pool
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

    // Fence for standalone copy submit
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(device_, &fci, nullptr, &copy_fence_));

    // NOTE: cuInit() is deferred to first CUDA operation (ensure_cuda_init)
}

void TensorTexture::destroy() {
    if (!device_) return;
    vkDeviceWaitIdle(device_);
    free_resources();
    if (copy_fence_) { vkDestroyFence(device_, copy_fence_, nullptr); copy_fence_ = VK_NULL_HANDLE; }
    if (cmd_pool_)   { vkDestroyCommandPool(device_, cmd_pool_, nullptr); cmd_pool_ = VK_NULL_HANDLE; }
    device_ = VK_NULL_HANDLE;
}

// ======================================================================
//  CUDA path: allocate_shared
// ======================================================================
#ifdef VULTORCH_HAS_CUDA
uintptr_t TensorTexture::allocate_shared(uint32_t width, uint32_t height,
                                         uint32_t channels, int cuda_device) {
    ensure_cuda_init();

    if (width == 0 || height == 0 || (channels != 1 && channels != 3 && channels != 4))
        throw std::runtime_error("TensorTexture::allocate_shared: invalid params");

    // Re-allocate only if dimensions changed or mode is wrong
    if (allocated_ && !staging_is_host_ && width_ == width && height_ == height && channels_ == channels) {
        return shared_cuda_ptr_;
    }

    free_resources();
    allocate_image(width, height);
    allocate_staging_external(width, height, channels, cuda_device);
    recreate_sampler();

    shared_cuda_ptr_ = static_cast<uintptr_t>(cuda_staging_ptr_);
    return shared_cuda_ptr_;
}

// ======================================================================
//  CUDA path: upload (GPU-GPU copy)
// ======================================================================
void TensorTexture::upload(uintptr_t data_ptr, uint32_t width, uint32_t height,
                           uint32_t channels, int cuda_device) {
    ensure_cuda_init();

    if (width == 0 || height == 0) return;
    if (channels != 1 && channels != 3 && channels != 4)
        throw std::runtime_error("TensorTexture: channels must be 1, 3, or 4");

    // Re-allocate if size changed or mode is wrong (was host-visible)
    if (!allocated_ || width_ != width || height_ != height
            || channels_ != channels || staging_is_host_) {
        free_resources();
        allocate_image(width, height);
        allocate_staging_external(width, height, channels, cuda_device);
        recreate_sampler();
    }

    // If the data_ptr IS our shared pointer, skip GPU-GPU memcpy
    if (is_shared_ptr(data_ptr)) {
        sync();
        return;
    }

    // Ensure CUDA context
    CUdevice cu_dev;
    CU_CHECK(cuDeviceGet(&cu_dev, cuda_device));
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    if (!ctx) CU_CHECK(cuCtxCreate(&ctx, 0, cu_dev));

    // GPU-GPU copy: source tensor → shared staging buffer.
    // Caller (Python show()) guarantees channels == 4 by padding on GPU.
    size_t src_bytes = (size_t)width * height * channels * sizeof(float);
    CU_CHECK(cuMemcpyDtoD(cuda_staging_ptr_, (CUdeviceptr)data_ptr, src_bytes));

    // Ensure CUDA copy is visible before Vulkan reads the staging buffer
    CU_CHECK(cuStreamSynchronize(nullptr));

    // Mark for deferred Vulkan copy (staging buffer → VkImage)
    needs_copy_ = true;
}

// ======================================================================
//  CUDA path: sync (zero-copy path)
// ======================================================================
void TensorTexture::sync() {
    if (!allocated_) return;

    if (channels_ != 4)
        throw std::runtime_error(
            "TensorTexture::sync() requires channels == 4 "
            "(Python show()/create_tensor() should pad to RGBA before calling)");

    CU_CHECK(cuStreamSynchronize(nullptr));
    needs_copy_ = true;
}

// ======================================================================
//  CUDA path: allocate staging buffer with external memory
// ======================================================================
void TensorTexture::allocate_staging_external(uint32_t w, uint32_t h,
                                              uint32_t ch, int cuda_device) {
    channels_ = ch;
    staging_is_host_ = false;

    // Always RGBA in the staging buffer for Vulkan image compatibility
    VkDeviceSize buf_size = (VkDeviceSize)w * h * 4 * sizeof(float);
    staging_size_ = buf_size;

    // ── VkBuffer ─────────────────────────────────────────────────
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = buf_size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    // External memory export info
    VkExternalMemoryBufferCreateInfo ext_buf_ci{};
    ext_buf_ci.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext_buf_ci.handleTypes = kExternalMemHandleType;
    bci.pNext = &ext_buf_ci;

    VK_CHECK(vkCreateBuffer(device_, &bci, nullptr, &staging_buf_));

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, staging_buf_, &memReq);

    // Dedicated allocation for external memory
    VkMemoryDedicatedAllocateInfo dedicated{};
    dedicated.sType  = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.buffer = staging_buf_;

    // Export info
    VkExportMemoryAllocateInfo export_ai{};
    export_ai.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    export_ai.handleTypes = kExternalMemHandleType;
    export_ai.pNext       = &dedicated;

    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = memReq.size;
    mai.memoryTypeIndex = find_memory_type(memReq.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    mai.pNext = &export_ai;

    VK_CHECK(vkAllocateMemory(device_, &mai, nullptr, &staging_mem_));
    VK_CHECK(vkBindBufferMemory(device_, staging_buf_, staging_mem_, 0));

    // ── Export handle ─────────────────────────────────────────────
#ifdef _WIN32
    VkMemoryGetWin32HandleInfoKHR handle_info{};
    handle_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handle_info.memory     = staging_mem_;
    handle_info.handleType = kExternalMemHandleType;

    auto vkGetMemoryWin32HandleKHR_ =
        (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandleKHR");
    if (!vkGetMemoryWin32HandleKHR_)
        throw std::runtime_error("vkGetMemoryWin32HandleKHR not available");

    HANDLE win32Handle = nullptr;
    VK_CHECK(vkGetMemoryWin32HandleKHR_(device_, &handle_info, &win32Handle));
#else
    VkMemoryGetFdInfoKHR fd_info{};
    fd_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fd_info.memory     = staging_mem_;
    fd_info.handleType = kExternalMemHandleType;

    auto vkGetMemoryFdKHR_ =
        (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR");
    if (!vkGetMemoryFdKHR_)
        throw std::runtime_error("vkGetMemoryFdKHR not available");

    int fd = -1;
    VK_CHECK(vkGetMemoryFdKHR_(device_, &fd_info, &fd));
#endif

    // ── CUDA import ───────────────────────────────────────────────
    CUdevice cu_dev;
    CU_CHECK(cuDeviceGet(&cu_dev, cuda_device));
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    if (!ctx) CU_CHECK(cuCtxCreate(&ctx, 0, cu_dev));

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC ext_desc{};
    ext_desc.size = memReq.size;
#ifdef _WIN32
    ext_desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
    ext_desc.handle.win32.handle = win32Handle;
#else
    ext_desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    ext_desc.handle.fd = fd;
#endif

    CU_CHECK(cuImportExternalMemory(&cuda_ext_mem_, &ext_desc));

    CUDA_EXTERNAL_MEMORY_BUFFER_DESC buf_desc{};
    buf_desc.offset = 0;
    buf_desc.size   = buf_size;
    CU_CHECK(cuExternalMemoryGetMappedBuffer(&cuda_staging_ptr_, cuda_ext_mem_, &buf_desc));

    allocated_ = true;
    std::cout << "[vultorch] TensorTexture: " << w << "x" << h << "x" << ch
              << " shared GPU memory allocated ("
              << (buf_size / 1024) << " KB)\n";
}
#endif // VULTORCH_HAS_CUDA

// ======================================================================
//  CPU path: upload from CPU memory
// ======================================================================
void TensorTexture::upload_cpu(const void* data, uint32_t width, uint32_t height,
                               uint32_t channels) {
    if (width == 0 || height == 0) return;
    if (channels != 4)
        throw std::runtime_error(
            "TensorTexture::upload_cpu: channels must be 4 (RGBA float32). "
            "Python show() pads to RGBA before calling.");

    // Re-allocate if size changed or mode is wrong (was external-memory)
    if (!allocated_ || width_ != width || height_ != height
            || channels_ != channels || !staging_is_host_) {
        free_resources();
        allocate_image(width, height);
        allocate_staging_host(width, height, channels);
        recreate_sampler();
    }

    // CPU memcpy to host-visible staging buffer (persistently mapped)
    size_t src_bytes = (size_t)width * height * channels * sizeof(float);
    std::memcpy(staging_mapped_, data, src_bytes);

    // Mark for deferred Vulkan copy (staging buffer → VkImage)
    needs_copy_ = true;
}

// ======================================================================
//  CPU path: allocate host-visible staging buffer
// ======================================================================
void TensorTexture::allocate_staging_host(uint32_t w, uint32_t h, uint32_t ch) {
    channels_ = ch;
    staging_is_host_ = true;

    // Always RGBA in staging for Vulkan image compatibility
    VkDeviceSize buf_size = (VkDeviceSize)w * h * 4 * sizeof(float);
    staging_size_ = buf_size;

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = buf_size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VK_CHECK(vkCreateBuffer(device_, &bci, nullptr, &staging_buf_));

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, staging_buf_, &memReq);

    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = memReq.size;
    mai.memoryTypeIndex = find_memory_type(memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(device_, &mai, nullptr, &staging_mem_));
    VK_CHECK(vkBindBufferMemory(device_, staging_buf_, staging_mem_, 0));

    // Persistently map for fast repeated uploads
    VK_CHECK(vkMapMemory(device_, staging_mem_, 0, buf_size, 0, &staging_mapped_));

    allocated_ = true;
    std::cout << "[vultorch] TensorTexture: " << w << "x" << h << "x" << ch
              << " host staging allocated ("
              << (buf_size / 1024) << " KB)\n";
}

// ======================================================================
//  Filter mode
// ======================================================================
void TensorTexture::set_filter(FilterMode mode) {
    if (mode == filter_ && sampler_) return;
    filter_ = mode;
    if (allocated_) recreate_sampler();
}

// ======================================================================
//  allocate_image — VkImage + VkImageView (RGBA float32, optimal tiling)
// ======================================================================
void TensorTexture::allocate_image(uint32_t w, uint32_t h) {
    width_  = w;
    height_ = h;

    VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;

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
    VK_CHECK(vkAllocateMemory(device_, &mai, nullptr, &image_mem_));
    VK_CHECK(vkBindImageMemory(device_, image_, image_mem_, 0));

    VkImageViewCreateInfo vci{};
    vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image    = image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = format;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device_, &vci, nullptr, &view_));
}

// ======================================================================
//  Sampler
// ======================================================================
void TensorTexture::recreate_sampler() {
    if (sampler_) {
        // Must remove old ImGui descriptor before destroying sampler
        if (descriptor_set_) {
            ImGui_ImplVulkan_RemoveTexture(descriptor_set_);
            descriptor_set_ = VK_NULL_HANDLE;
        }
        vkDestroySampler(device_, sampler_, nullptr);
        sampler_ = VK_NULL_HANDLE;
    }

    VkFilter vk_filter = (filter_ == FilterMode::Nearest)
                             ? VK_FILTER_NEAREST
                             : VK_FILTER_LINEAR;

    VkSamplerCreateInfo si{};
    si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter    = vk_filter;
    si.minFilter    = vk_filter;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device_, &si, nullptr, &sampler_));

    // Register with ImGui
    descriptor_set_ = ImGui_ImplVulkan_AddTexture(
        sampler_, view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

// ======================================================================
//  record_copy_commands — barrier + copy + barrier (records into any cmd buf)
// ======================================================================
void TensorTexture::record_copy_commands(VkCommandBuffer cmd) {
    // ── Barrier: UNDEFINED → TRANSFER_DST ────────────────────────
    VkImageMemoryBarrier b1{};
    b1.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b1.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    b1.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.image               = image_;
    b1.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b1.srcAccessMask       = 0;
    b1.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &b1);

    // ── Copy buffer → image ──────────────────────────────────────
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {width_, height_, 1};
    vkCmdCopyBufferToImage(cmd, staging_buf_, image_,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // ── Barrier: TRANSFER_DST → SHADER_READ_ONLY ─────────────────
    VkImageMemoryBarrier b2{};
    b2.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b2.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b2.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b2.image               = image_;
    b2.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b2.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    b2.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &b2);
}

// ======================================================================
//  copy_staging_to_image — standalone submit with VkFence
// ======================================================================
void TensorTexture::copy_staging_to_image() {
    if (!needs_copy_ || !allocated_) return;

    // Wait for any previous standalone copy to complete
    if (fence_pending_) {
        VK_CHECK(vkWaitForFences(device_, 1, &copy_fence_, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(device_, 1, &copy_fence_));
        fence_pending_ = false;
    }

    needs_copy_ = false;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkResetCommandBuffer(cmd_buf_, 0));
    VK_CHECK(vkBeginCommandBuffer(cmd_buf_, &bi));

    record_copy_commands(cmd_buf_);

    VK_CHECK(vkEndCommandBuffer(cmd_buf_));

    // Submit with fence — no vkQueueWaitIdle!
    VkSubmitInfo submit{};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd_buf_;
    VK_CHECK(vkQueueSubmit(queue_, 1, &submit, copy_fence_));
    fence_pending_ = true;
}

// ======================================================================
//  record_copy — integrate copy into an external command buffer
// ======================================================================
void TensorTexture::record_copy(VkCommandBuffer cmd) {
    if (!needs_copy_ || !allocated_) return;

    // If there's a pending standalone fence, wait for it
    if (fence_pending_) {
        VK_CHECK(vkWaitForFences(device_, 1, &copy_fence_, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(device_, 1, &copy_fence_));
        fence_pending_ = false;
    }

    needs_copy_ = false;
    record_copy_commands(cmd);
}

// ======================================================================
//  wait_for_copy / flush
// ======================================================================
void TensorTexture::wait_for_copy() {
    if (!fence_pending_) return;
    VK_CHECK(vkWaitForFences(device_, 1, &copy_fence_, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(device_, 1, &copy_fence_));
    fence_pending_ = false;
}

void TensorTexture::flush() {
    if (needs_copy_) copy_staging_to_image();
    wait_for_copy();
}

// ======================================================================
//  free_resources
// ======================================================================
void TensorTexture::free_resources() {
    if (!allocated_) return;
    vkDeviceWaitIdle(device_);
    fence_pending_ = false;
    needs_copy_ = false;

#ifdef VULTORCH_HAS_CUDA
    // CUDA side (external-memory path only)
    if (cuda_staging_ptr_) { cuda_staging_ptr_ = 0; }
    if (cuda_ext_mem_)     { cuDestroyExternalMemory(cuda_ext_mem_); cuda_ext_mem_ = nullptr; }
    shared_cuda_ptr_ = 0;
#endif

    // Unmap host staging if mapped
    if (staging_mapped_) {
        vkUnmapMemory(device_, staging_mem_);
        staging_mapped_ = nullptr;
    }

    // ImGui descriptor
    if (descriptor_set_) {
        ImGui_ImplVulkan_RemoveTexture(descriptor_set_);
        descriptor_set_ = VK_NULL_HANDLE;
    }

    // Staging
    if (staging_buf_) { vkDestroyBuffer(device_, staging_buf_, nullptr); staging_buf_ = VK_NULL_HANDLE; }
    if (staging_mem_) { vkFreeMemory(device_, staging_mem_, nullptr);    staging_mem_ = VK_NULL_HANDLE; }

    // Image
    if (sampler_)   { vkDestroySampler(device_, sampler_, nullptr);     sampler_   = VK_NULL_HANDLE; }
    if (view_)      { vkDestroyImageView(device_, view_, nullptr);      view_      = VK_NULL_HANDLE; }
    if (image_)     { vkDestroyImage(device_, image_, nullptr);         image_     = VK_NULL_HANDLE; }
    if (image_mem_) { vkFreeMemory(device_, image_mem_, nullptr);       image_mem_ = VK_NULL_HANDLE; }

    staging_is_host_ = false;
    allocated_ = false;
}

// ======================================================================
//  Helpers
// ======================================================================
uint32_t TensorTexture::find_memory_type(uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(phys_device_, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
        if ((filter & (1u << i)) && (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("TensorTexture: failed to find suitable memory type");
}

} // namespace vultorch
