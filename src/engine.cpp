#include "engine.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>

namespace vultorch {

namespace {
int g_sdl_refcount = 0;

/// Human-readable Vulkan result codes for error messages.
const char* vk_result_string(VkResult r) {
    switch (r) {
        case VK_SUCCESS:                        return "VK_SUCCESS";
        case VK_NOT_READY:                      return "VK_NOT_READY";
        case VK_TIMEOUT:                        return "VK_TIMEOUT";
        case VK_ERROR_OUT_OF_HOST_MEMORY:       return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:     return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED:    return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST:              return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED:        return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT:        return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT:    return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT:      return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_TOO_MANY_OBJECTS:         return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED:     return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_SURFACE_LOST_KHR:         return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR:          return "VK_ERROR_OUT_OF_DATE_KHR";
        case VK_SUBOPTIMAL_KHR:                 return "VK_SUBOPTIMAL_KHR";
        default:                                return "VK_UNKNOWN_ERROR";
    }
}
} // anonymous namespace

// ---------------------------------------------------------------------------
// Vulkan error-check helper
// ---------------------------------------------------------------------------
#define VK_CHECK(x)                                                             \
    do {                                                                        \
        VkResult _r = (x);                                                      \
        if (_r != VK_SUCCESS)                                                   \
            throw std::runtime_error(                                           \
                std::string("[vultorch] Vulkan error: ") +                      \
                vk_result_string(_r) + " (" +                                  \
                std::to_string(static_cast<int>(_r)) + ") at " +               \
                __FILE__ + ":" + std::to_string(__LINE__));                     \
    } while (0)

// ===================================================================== dtor
Engine::~Engine() {
    if (initialized_) destroy();
}

// ==================================================================== init
void Engine::init(const char* title, int width, int height) {
    if (g_sdl_refcount == 0) {
        if (!SDL_Init(SDL_INIT_VIDEO))
            throw std::runtime_error(
                std::string("[vultorch] SDL_Init failed: ") + SDL_GetError());
    }
    g_sdl_refcount++;

    window_ = SDL_CreateWindow(
        title, width, height,
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    if (!window_)
        throw std::runtime_error(
            std::string("[vultorch] SDL_CreateWindow failed: ") + SDL_GetError());

    create_instance();
    create_surface();
    pick_physical_device();
    create_device();
    create_swapchain();
    create_render_pass();
    create_framebuffers();
    create_commands();
    create_sync();
    init_imgui();

    initialized_ = true;
}

TensorTexture& Engine::tensor_texture(const std::string& key) {
    auto it = tensor_textures_.find(key);
    if (it == tensor_textures_.end()) {
        auto tex = std::make_unique<TensorTexture>();
        tex->init(instance_, physical_device_, device_,
                  graphics_queue_, graphics_family_,
                  imgui_pool_);
        it = tensor_textures_.emplace(key, std::move(tex)).first;
    }
    return *it->second;
}

SceneRenderer& Engine::scene_renderer(uint32_t width, uint32_t height,
                                      VkSampleCountFlagBits msaa) {
    if (!scene_renderer_) {
        scene_renderer_ = std::make_unique<SceneRenderer>();
        scene_renderer_->init(instance_, physical_device_, device_,
                              graphics_queue_, graphics_family_,
                              imgui_pool_, width, height, msaa);
    }
    return *scene_renderer_;
}

VkSampleCountFlagBits Engine::max_msaa_samples() const {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    VkSampleCountFlags counts = props.limits.framebufferColorSampleCounts &
                                props.limits.framebufferDepthSampleCounts;
    if (counts & VK_SAMPLE_COUNT_8_BIT) return VK_SAMPLE_COUNT_8_BIT;
    if (counts & VK_SAMPLE_COUNT_4_BIT) return VK_SAMPLE_COUNT_4_BIT;
    if (counts & VK_SAMPLE_COUNT_2_BIT) return VK_SAMPLE_COUNT_2_BIT;
    return VK_SAMPLE_COUNT_1_BIT;
}

// ================================================================= destroy
void Engine::destroy() {
    if (!initialized_) return;
    vkDeviceWaitIdle(device_);

    scene_renderer_.reset();
    tensor_textures_.clear();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    if (imgui_pool_) vkDestroyDescriptorPool(device_, imgui_pool_, nullptr);

    for (uint32_t i = 0; i < MAX_FRAMES; i++) {
        vkDestroySemaphore(device_, sem_available_[i], nullptr);
        vkDestroySemaphore(device_, sem_finished_[i], nullptr);
        vkDestroyFence(device_, fences_[i], nullptr);
    }

    vkDestroyCommandPool(device_, command_pool_, nullptr);
    cleanup_swapchain();
    vkDestroyRenderPass(device_, render_pass_, nullptr);
    vkDestroyDevice(device_, nullptr);
    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    vkDestroyInstance(instance_, nullptr);

    if (window_) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }

    if (g_sdl_refcount > 0) {
        g_sdl_refcount--;
        if (g_sdl_refcount == 0)
            SDL_Quit();
    }

    initialized_ = false;
}

// ==================================================================== poll
bool Engine::poll() {
    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
        ImGui_ImplSDL3_ProcessEvent(&ev);
        if (ev.type == SDL_EVENT_QUIT) return false;
        if (ev.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED)
            return false;
    }
    return true;
}

// ============================================================= begin_frame
bool Engine::begin_frame() {
    // Skip if minimised (0×0)
    int w, h;
    SDL_GetWindowSize(window_, &w, &h);
    if (w == 0 || h == 0) { frame_started_ = false; return false; }

    VK_CHECK(vkWaitForFences(device_, 1, &fences_[frame_index_], VK_TRUE, UINT64_MAX));

    VkResult result = vkAcquireNextImageKHR(
        device_, swapchain_, UINT64_MAX,
        sem_available_[frame_index_], VK_NULL_HANDLE, &image_index_);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreate_swapchain();
        frame_started_ = false;
        return false;
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error(
            std::string("[vultorch] vkAcquireNextImageKHR failed: ") +
            vk_result_string(result));

    VK_CHECK(vkResetFences(device_, 1, &fences_[frame_index_]));
    VK_CHECK(vkResetCommandBuffer(command_buffers_[frame_index_], 0));

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    frame_started_ = true;
    return true;
}

// =============================================================== end_frame
void Engine::end_frame() {
    if (!frame_started_) return;

    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();

    VkCommandBuffer cmd = command_buffers_[frame_index_];

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin_info));

    // Integrate pending tensor copies into the main command buffer.
    // This avoids separate vkQueueSubmit calls; pipeline barriers inside
    // record_copy() ensure images are in SHADER_READ_ONLY before
    // the render pass samples them.
    for (auto& kv : tensor_textures_) {
        kv.second->record_copy(cmd);
    }

    // Record offscreen 3D scene render (if requested this frame).
    // This replaces the old standalone submit + vkQueueWaitIdle path.
    if (scene_renderer_ && scene_renderer_->needs_render()) {
        scene_renderer_->record_render(cmd);

        // Barrier: ensure offscreen resolve writes are visible to
        // fragment shader reads in the main (ImGui) render pass.
        VkMemoryBarrier barrier{};
        barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    VkClearValue clear{};
    clear.color = {{0.08f, 0.08f, 0.10f, 1.0f}};

    VkRenderPassBeginInfo rp_info{};
    rp_info.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_info.renderPass        = render_pass_;
    rp_info.framebuffer       = framebuffers_[image_index_];
    rp_info.renderArea.extent = swapchain_extent_;
    rp_info.clearValueCount   = 1;
    rp_info.pClearValues      = &clear;

    vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(draw_data, cmd);
    vkCmdEndRenderPass(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    // Submit
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo submit{};
    submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount   = 1;
    submit.pWaitSemaphores      = &sem_available_[frame_index_];
    submit.pWaitDstStageMask    = &wait_stage;
    submit.commandBufferCount   = 1;
    submit.pCommandBuffers      = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores    = &sem_finished_[frame_index_];

    VK_CHECK(vkQueueSubmit(graphics_queue_, 1, &submit, fences_[frame_index_]));

    // Present
    VkPresentInfoKHR present{};
    present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores    = &sem_finished_[frame_index_];
    present.swapchainCount     = 1;
    present.pSwapchains        = &swapchain_;
    present.pImageIndices      = &image_index_;

    VkResult result = vkQueuePresentKHR(graphics_queue_, &present);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        recreate_swapchain();

    frame_index_ = (frame_index_ + 1) % MAX_FRAMES;
    frame_started_ = false;
}

// =====================================================================
//  Vulkan initialisation helpers
// =====================================================================

void Engine::create_instance() {
    VkApplicationInfo app{};
    app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName   = "Vultorch";
    app.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app.pEngineName        = "Vultorch";
    app.engineVersion      = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app.apiVersion         = VK_API_VERSION_1_2;

    // Extensions required by SDL
    uint32_t sdl_count = 0;
    const char* const* sdl_exts = SDL_Vulkan_GetInstanceExtensions(&sdl_count);
    std::vector<const char*> extensions(sdl_exts, sdl_exts + sdl_count);

    VkInstanceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo        = &app;
    ci.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();
    ci.enabledLayerCount       = 0;

    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));
}

void Engine::create_surface() {
    if (!SDL_Vulkan_CreateSurface(window_, instance_, nullptr, &surface_))
        throw std::runtime_error("[vultorch] SDL_Vulkan_CreateSurface failed");
}

void Engine::pick_physical_device() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) throw std::runtime_error("[vultorch] No Vulkan-capable GPU found");

    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(instance_, &count, devs.data());

    // Score each device: prefer discrete GPU, then integrated, then others.
    struct Candidate {
        VkPhysicalDevice device;
        uint32_t         queue_family;
        int              score;
        std::string      name;
    };
    std::vector<Candidate> candidates;

    for (auto& dev : devs) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);

        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, qprops.data());

        for (uint32_t i = 0; i < qcount; i++) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface_, &present);
            if ((qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                int score = 0;
                switch (props.deviceType) {
                    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   score = 1000; break;
                    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score = 100;  break;
                    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    score = 50;   break;
                    default:                                     score = 10;   break;
                }
                candidates.push_back({dev, i, score, props.deviceName});
                break; // one candidate per device
            }
        }
    }

    if (candidates.empty())
        throw std::runtime_error(
            "[vultorch] No suitable GPU found (need graphics + present support)");

    // Pick the highest-scoring device
    auto best = std::max_element(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.score < b.score; });

    physical_device_ = best->device;
    graphics_family_ = best->queue_family;

    std::cout << "[vultorch] Selected GPU: " << best->name << "\n";
    if (candidates.size() > 1) {
        std::cout << "[vultorch] Other available GPUs:\n";
        for (auto& c : candidates) {
            if (&c != &(*best))
                std::cout << "[vultorch]   - " << c.name << "\n";
        }
    }
}

void Engine::create_device() {
    float priority = 1.0f;

    VkDeviceQueueCreateInfo queue_ci{};
    queue_ci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_ci.queueFamilyIndex = graphics_family_;
    queue_ci.queueCount       = 1;
    queue_ci.pQueuePriorities = &priority;

    std::vector<const char*> dev_exts = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

#ifdef VULTORCH_HAS_CUDA
    // External memory extensions for CUDA interop
#ifdef _WIN32
    dev_exts.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    dev_exts.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
    dev_exts.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    dev_exts.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif
#endif

    VkDeviceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.queueCreateInfoCount    = 1;
    ci.pQueueCreateInfos       = &queue_ci;
    ci.enabledExtensionCount   = static_cast<uint32_t>(dev_exts.size());
    ci.ppEnabledExtensionNames = dev_exts.data();

    VK_CHECK(vkCreateDevice(physical_device_, &ci, nullptr, &device_));
    vkGetDeviceQueue(device_, graphics_family_, 0, &graphics_queue_);
}

void Engine::create_swapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface_, &caps);

    // Format
    uint32_t fmt_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &fmt_count, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmt_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &fmt_count, fmts.data());

    swapchain_format_ = fmts[0].format;
    VkColorSpaceKHR color_space = fmts[0].colorSpace;
    for (auto& f : fmts) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            swapchain_format_ = f.format;
            color_space = f.colorSpace;
            break;
        }
    }

    // Extent
    if (caps.currentExtent.width != (std::numeric_limits<uint32_t>::max)()) {
        swapchain_extent_ = caps.currentExtent;
    } else {
        int w, h;
        SDL_GetWindowSizeInPixels(window_, &w, &h);
        swapchain_extent_.width  = std::clamp(static_cast<uint32_t>(w),
                                              caps.minImageExtent.width,
                                              caps.maxImageExtent.width);
        swapchain_extent_.height = std::clamp(static_cast<uint32_t>(h),
                                              caps.minImageExtent.height,
                                              caps.maxImageExtent.height);
    }

    uint32_t img_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && img_count > caps.maxImageCount)
        img_count = caps.maxImageCount;

    VkSwapchainCreateInfoKHR ci{};
    ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface          = surface_;
    ci.minImageCount    = img_count;
    ci.imageFormat      = swapchain_format_;
    ci.imageColorSpace  = color_space;
    ci.imageExtent      = swapchain_extent_;
    ci.imageArrayLayers = 1;
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform     = caps.currentTransform;
    ci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode      = VK_PRESENT_MODE_FIFO_KHR;
    ci.clipped          = VK_TRUE;

    VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));

    // Swapchain images
    vkGetSwapchainImagesKHR(device_, swapchain_, &img_count, nullptr);
    swapchain_images_.resize(img_count);
    vkGetSwapchainImagesKHR(device_, swapchain_, &img_count, swapchain_images_.data());

    // Image views
    swapchain_views_.resize(img_count);
    for (uint32_t i = 0; i < img_count; i++) {
        VkImageViewCreateInfo vi{};
        vi.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image    = swapchain_images_[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format   = swapchain_format_;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &swapchain_views_[i]));
    }
}

void Engine::create_render_pass() {
    VkAttachmentDescription color{};
    color.format         = swapchain_format_;
    color.samples        = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference ref{};
    ref.attachment = 0;
    ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &ref;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo ci{};
    ci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.attachmentCount = 1;
    ci.pAttachments    = &color;
    ci.subpassCount    = 1;
    ci.pSubpasses      = &subpass;
    ci.dependencyCount = 1;
    ci.pDependencies   = &dep;

    VK_CHECK(vkCreateRenderPass(device_, &ci, nullptr, &render_pass_));
}

void Engine::create_framebuffers() {
    framebuffers_.resize(swapchain_views_.size());
    for (size_t i = 0; i < swapchain_views_.size(); i++) {
        VkFramebufferCreateInfo ci{};
        ci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        ci.renderPass      = render_pass_;
        ci.attachmentCount = 1;
        ci.pAttachments    = &swapchain_views_[i];
        ci.width           = swapchain_extent_.width;
        ci.height          = swapchain_extent_.height;
        ci.layers          = 1;
        VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &framebuffers_[i]));
    }
}

void Engine::create_commands() {
    VkCommandPoolCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = graphics_family_;
    VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &command_pool_));

    command_buffers_.resize(MAX_FRAMES);
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = command_pool_;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = MAX_FRAMES;
    VK_CHECK(vkAllocateCommandBuffers(device_, &ai, command_buffers_.data()));
}

void Engine::create_sync() {
    sem_available_.resize(MAX_FRAMES);
    sem_finished_.resize(MAX_FRAMES);
    fences_.resize(MAX_FRAMES);

    VkSemaphoreCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (uint32_t i = 0; i < MAX_FRAMES; i++) {
        VK_CHECK(vkCreateSemaphore(device_, &si, nullptr, &sem_available_[i]));
        VK_CHECK(vkCreateSemaphore(device_, &si, nullptr, &sem_finished_[i]));
        VK_CHECK(vkCreateFence(device_, &fi, nullptr, &fences_[i]));
    }
}

void Engine::init_imgui() {
    // Descriptor pool for ImGui
    VkDescriptorPoolSize pool_size{};
    pool_size.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_size.descriptorCount = 100;

    VkDescriptorPoolCreateInfo pi{};
    pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pi.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pi.maxSets       = 100;
    pi.poolSizeCount = 1;
    pi.pPoolSizes    = &pool_size;
    VK_CHECK(vkCreateDescriptorPool(device_, &pi, nullptr, &imgui_pool_));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();

    ImGui_ImplSDL3_InitForVulkan(window_);

    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion     = VK_API_VERSION_1_2;
    info.Instance       = instance_;
    info.PhysicalDevice = physical_device_;
    info.Device         = device_;
    info.QueueFamily    = graphics_family_;
    info.Queue          = graphics_queue_;
    info.DescriptorPool = imgui_pool_;
    info.MinImageCount  = static_cast<uint32_t>(swapchain_images_.size());
    info.ImageCount     = static_cast<uint32_t>(swapchain_images_.size());
    info.PipelineInfoMain.RenderPass  = render_pass_;
    info.PipelineInfoMain.Subpass     = 0;
    info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&info);
    // Font atlas auto-created by NewFrame() in imgui 1.92+
}

void Engine::cleanup_swapchain() {
    for (auto fb : framebuffers_)      vkDestroyFramebuffer(device_, fb, nullptr);
    for (auto iv : swapchain_views_)   vkDestroyImageView(device_, iv, nullptr);
    if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    framebuffers_.clear();
    swapchain_views_.clear();
    swapchain_images_.clear();
    swapchain_ = VK_NULL_HANDLE;
}

void Engine::recreate_swapchain() {
    int w = 0, h = 0;
    SDL_GetWindowSize(window_, &w, &h);
    while (w == 0 || h == 0) {
        SDL_GetWindowSize(window_, &w, &h);
        SDL_WaitEvent(nullptr);
    }
    vkDeviceWaitIdle(device_);
    cleanup_swapchain();
    create_swapchain();
    create_framebuffers();
}

} // namespace vultorch
