// TensorTexture — Vulkan texture for tensor display (CUDA + CPU paths).
// See tensor_texture.h for the three usage modes.

#include "tensor_texture.h"
#include "vk_utils.h"
#include "log.h"
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector>
#include <algorithm>

// Embedded SPIR-V for float32→R8G8B8A8_UNORM compute shader
#include "float_to_unorm_spv.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace vultorch {

#ifdef VULTORCH_HAS_CUDA
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
                         VkDescriptorPool imguiPool,
                         VkCommandPool external_pool) {
    instance_     = instance;
    phys_device_  = physDevice;
    device_       = device;
    queue_        = queue;
    queue_family_ = queueFamily;
    imgui_pool_   = imguiPool;

    // Use external pool if provided, otherwise create our own
    if (external_pool != VK_NULL_HANDLE) {
        cmd_pool_  = external_pool;
        owns_pool_ = false;
    } else {
        VkCommandPoolCreateInfo ci{};
        ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueFamily;
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &cmd_pool_));
        owns_pool_ = true;
    }

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
    wait_for_copy();  // wait for our own pending copy fence (no full GPU stall)
    free_resources();
    if (copy_fence_) { vkDestroyFence(device_, copy_fence_, nullptr); copy_fence_ = VK_NULL_HANDLE; }
    if (cmd_buf_ && cmd_pool_) {
        vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd_buf_);
        cmd_buf_ = VK_NULL_HANDLE;
    }
    if (cmd_pool_ && owns_pool_) {
        vkDestroyCommandPool(device_, cmd_pool_, nullptr);
    }
    cmd_pool_ = VK_NULL_HANDLE;
    owns_pool_ = false;
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
    create_compute_descriptor();

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
        create_compute_descriptor();
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
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    // External memory export info
    VkExternalMemoryBufferCreateInfo ext_buf_ci{};
    ext_buf_ci.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext_buf_ci.handleTypes = kExternalMemHandleType;
    bci.pNext = &ext_buf_ci;

    VkBuffer raw_buf;
    VK_CHECK(vkCreateBuffer(device_, &bci, nullptr, &raw_buf));
    staging_buf_.reset(device_, raw_buf);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, raw_buf, &memReq);

    // Dedicated allocation for external memory
    VkMemoryDedicatedAllocateInfo dedicated{};
    dedicated.sType  = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated.buffer = raw_buf;

    // Export info
    VkExportMemoryAllocateInfo export_ai{};
    export_ai.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    export_ai.handleTypes = kExternalMemHandleType;
    export_ai.pNext       = &dedicated;

    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = memReq.size;
    mai.memoryTypeIndex = vultorch::find_memory_type(phys_device_, memReq.memoryTypeBits,
                                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    mai.pNext = &export_ai;

    VkDeviceMemory raw_mem;
    VK_CHECK(vkAllocateMemory(device_, &mai, nullptr, &raw_mem));
    staging_mem_.reset(device_, raw_mem);
    VK_CHECK(vkBindBufferMemory(device_, raw_buf, raw_mem, 0));

    // ── Export handle ─────────────────────────────────────────────
#ifdef _WIN32
    VkMemoryGetWin32HandleInfoKHR handle_info{};
    handle_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handle_info.memory     = raw_mem;
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
    fd_info.memory     = raw_mem;
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
    VT_INFO("TensorTexture: " << w << "x" << h << "x" << ch
              << " shared GPU memory allocated ("
              << (buf_size / 1024) << " KB)");
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
        create_compute_descriptor();
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
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBuffer raw_buf;
    VK_CHECK(vkCreateBuffer(device_, &bci, nullptr, &raw_buf));

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, raw_buf, &memReq);

    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = memReq.size;
    mai.memoryTypeIndex = vultorch::find_memory_type(phys_device_, memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory raw_mem;
    VK_CHECK(vkAllocateMemory(device_, &mai, nullptr, &raw_mem));
    VK_CHECK(vkBindBufferMemory(device_, raw_buf, raw_mem, 0));

    // Persistently map for fast repeated uploads
    VK_CHECK(vkMapMemory(device_, raw_mem, 0, buf_size, 0, &staging_mapped_));

    staging_buf_.reset(device_, raw_buf);
    staging_mem_.reset(device_, raw_mem);

    allocated_ = true;
    VT_INFO("TensorTexture: " << w << "x" << h << "x" << ch
              << " host staging allocated ("
              << (buf_size / 1024) << " KB)");
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
//  allocate_image — VkImage + VkImageView (R8G8B8A8_UNORM, optimal tiling)
//  Uses compute shader to convert float32 staging → uint8 image (75% VRAM saving)
// ======================================================================
void TensorTexture::allocate_image(uint32_t w, uint32_t h) {
    width_  = w;
    height_ = h;

    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = format;
    ici.extent        = {w, h, 1};
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage raw_img;
    VK_CHECK(vkCreateImage(device_, &ici, nullptr, &raw_img));

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device_, raw_img, &memReq);

    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = memReq.size;
    mai.memoryTypeIndex = vultorch::find_memory_type(phys_device_, memReq.memoryTypeBits,
                                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory raw_mem;
    VK_CHECK(vkAllocateMemory(device_, &mai, nullptr, &raw_mem));
    VK_CHECK(vkBindImageMemory(device_, raw_img, raw_mem, 0));

    image_.reset(device_, raw_img);
    image_mem_.reset(device_, raw_mem);

    VkImageViewCreateInfo vci{};
    vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image    = raw_img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = format;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.layerCount = 1;
    VkImageView raw_view;
    VK_CHECK(vkCreateImageView(device_, &vci, nullptr, &raw_view));
    view_.reset(device_, raw_view);

    // Create compute pipeline (reusable across re-allocations)
    create_compute_pipeline();
    // Note: compute descriptor is created after staging buffer allocation
    // (needs both image view and staging buffer)
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
        sampler_.reset();
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
    VkSampler raw;
    VK_CHECK(vkCreateSampler(device_, &si, nullptr, &raw));
    sampler_.reset(device_, raw);

    // Register with ImGui
    descriptor_set_ = ImGui_ImplVulkan_AddTexture(
        sampler_.get(), view_.get(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

// ======================================================================
//  Compute pipeline for float32 → R8G8B8A8_UNORM conversion
// ======================================================================
void TensorTexture::create_compute_pipeline() {
    if (compute_pipeline_) return;  // already created

    // Shader module from embedded SPIR-V (RAII — auto-destroyed at scope exit)
    VkShaderModuleCreateInfo smi{};
    smi.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = float_to_unorm_spv_size;
    smi.pCode    = float_to_unorm_spv;
    VkShaderModule raw_shader;
    VK_CHECK(vkCreateShaderModule(device_, &smi, nullptr, &raw_shader));
    VkUniqueShaderModule shader(device_, raw_shader);

    // Descriptor set layout: binding 0 = storage buffer, binding 1 = storage image
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dsl{};
    dsl.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl.bindingCount = 2;
    dsl.pBindings    = bindings;
    VkDescriptorSetLayout raw_dsl;
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &dsl, nullptr, &raw_dsl));
    compute_desc_layout_.reset(device_, raw_dsl);

    // Push constant range: {width, height}
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(uint32_t) * 2;

    VkPipelineLayoutCreateInfo pli{};
    pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount         = 1;
    pli.pSetLayouts            = compute_desc_layout_.ptr();
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges    = &pcr;
    VkPipelineLayout raw_pl;
    VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &raw_pl));
    compute_layout_.reset(device_, raw_pl);

    // Compute pipeline
    VkComputePipelineCreateInfo cpi{};
    cpi.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = shader;
    cpi.stage.pName  = "main";
    cpi.layout = compute_layout_;
    VkPipeline raw_pipe;
    VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpi, nullptr, &raw_pipe));
    compute_pipeline_.reset(device_, raw_pipe);

    // shader auto-destroyed by VkUniqueShaderModule
}

void TensorTexture::create_compute_descriptor() {
    // Recreate pool + set each time image/staging changes
    if (compute_desc_) {
        // Pool reset frees all sets allocated from it
        vkResetDescriptorPool(device_, compute_pool_.get(), 0);
        compute_desc_ = VK_NULL_HANDLE;
    }
    if (!compute_pool_) {
        VkDescriptorPoolSize sizes[2]{};
        sizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        sizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1};

        VkDescriptorPoolCreateInfo pi{};
        pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.maxSets       = 1;
        pi.poolSizeCount = 2;
        pi.pPoolSizes    = sizes;
        VkDescriptorPool raw_pool;
        VK_CHECK(vkCreateDescriptorPool(device_, &pi, nullptr, &raw_pool));
        compute_pool_.reset(device_, raw_pool);
    }

    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = compute_pool_.get();
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = compute_desc_layout_.ptr();
    VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &compute_desc_));

    // Update: binding 0 = staging buffer, binding 1 = destination image
    VkDescriptorBufferInfo buf_info{staging_buf_.get(), 0, staging_size_};
    VkDescriptorImageInfo  img_info{VK_NULL_HANDLE, view_.get(), VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = compute_desc_;
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo     = &buf_info;

    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = compute_desc_;
    writes[1].dstBinding      = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo      = &img_info;

    vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
}

// ======================================================================
//  record_copy_commands — compute dispatch: float32 staging → R8G8B8A8 image
// ======================================================================
void TensorTexture::record_copy_commands(VkCommandBuffer cmd) {
    // ── Barrier: image UNDEFINED → GENERAL (for compute imageStore) ──
    VkImageMemoryBarrier b1{};
    b1.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b1.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    b1.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
    b1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.image               = image_;
    b1.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b1.srcAccessMask       = 0;
    b1.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &b1);

    // ── Dispatch compute shader ──────────────────────────────────
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute_layout_, 0, 1, &compute_desc_, 0, nullptr);
    uint32_t pc[2] = {width_, height_};
    vkCmdPushConstants(cmd, compute_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(cmd, (width_ + 15) / 16, (height_ + 15) / 16, 1);

    // ── Barrier: image GENERAL → SHADER_READ_ONLY ────────────────
    VkImageMemoryBarrier b2{};
    b2.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b2.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
    b2.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b2.image               = image_;
    b2.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b2.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    b2.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
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
    wait_for_copy();  // wait for our own pending copy fence (no full GPU stall)
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
        vkUnmapMemory(device_, staging_mem_.get());
        staging_mapped_ = nullptr;
    }

    // ImGui descriptor (managed by ImGui, not RAII)
    if (descriptor_set_) {
        ImGui_ImplVulkan_RemoveTexture(descriptor_set_);
        descriptor_set_ = VK_NULL_HANDLE;
    }

    // Staging
    staging_buf_.reset();
    staging_mem_.reset();

    // Image
    sampler_.reset();
    view_.reset();
    image_.reset();
    image_mem_.reset();

    // Compute pipeline resources
    compute_desc_ = VK_NULL_HANDLE;  // freed with pool
    compute_pool_.reset();
    compute_pipeline_.reset();
    compute_layout_.reset();
    compute_desc_layout_.reset();

    staging_is_host_ = false;
    allocated_ = false;
}

} // namespace vultorch
