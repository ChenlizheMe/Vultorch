#include "scene_renderer.h"
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <array>
#include <algorithm>

// ── Embedded SPIR-V (generated at build time) ─────────────────────────
#include "plane_vert_spv.h"
#include "plane_frag_spv.h"

namespace vultorch {

#define VK_CHECK(x)                                                           \
    do {                                                                      \
        VkResult _r = (x);                                                    \
        if (_r != VK_SUCCESS)                                                 \
            throw std::runtime_error(                                         \
                std::string("Vulkan error ") + std::to_string((int)_r) +      \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
    } while (0)

// ======================================================================
SceneRenderer::~SceneRenderer() { destroy(); }

void SceneRenderer::init(VkInstance instance, VkPhysicalDevice physDevice,
                         VkDevice device, VkQueue queue, uint32_t queueFamily,
                         VkDescriptorPool imguiPool,
                         uint32_t width, uint32_t height,
                         VkSampleCountFlagBits msaa) {
    instance_      = instance;
    phys_device_   = physDevice;
    device_        = device;
    queue_         = queue;
    queue_family_  = queueFamily;
    imgui_pool_    = imguiPool;
    width_         = width;
    height_        = height;
    msaa_samples_  = msaa;

    create_offscreen_resources();
    create_render_pass();
    create_framebuffer();
    create_plane_mesh();
    create_ubos();
    // Pipeline & descriptor set created lazily on first render() call
    // because we need the TensorTexture's sampler/view

    initialized_ = true;
    std::cout << "[vultorch] SceneRenderer: " << width << "x" << height
              << " MSAA " << (int)msaa << "x\n";
}

void SceneRenderer::destroy() {
    if (!initialized_) return;
    vkDeviceWaitIdle(device_);

    // ImGui descriptor
    if (imgui_desc_) {
        ImGui_ImplVulkan_RemoveTexture(imgui_desc_);
        imgui_desc_ = VK_NULL_HANDLE;
    }
    if (resolve_sampler_) { vkDestroySampler(device_, resolve_sampler_, nullptr); resolve_sampler_ = VK_NULL_HANDLE; }

    // Pipeline
    if (pipeline_)        { vkDestroyPipeline(device_, pipeline_, nullptr);             pipeline_ = VK_NULL_HANDLE; }
    if (pipeline_layout_) { vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr); pipeline_layout_ = VK_NULL_HANDLE; }
    if (desc_layout_)     { vkDestroyDescriptorSetLayout(device_, desc_layout_, nullptr); desc_layout_ = VK_NULL_HANDLE; }
    if (scene_pool_)      { vkDestroyDescriptorPool(device_, scene_pool_, nullptr);       scene_pool_ = VK_NULL_HANDLE; }

    // UBOs
    if (ubo_mapped_)      { vkUnmapMemory(device_, ubo_mem_);       ubo_mapped_ = nullptr; }
    if (ubo_buf_)         { vkDestroyBuffer(device_, ubo_buf_, nullptr);  ubo_buf_ = VK_NULL_HANDLE; }
    if (ubo_mem_)         { vkFreeMemory(device_, ubo_mem_, nullptr);     ubo_mem_ = VK_NULL_HANDLE; }
    if (light_ubo_mapped_) { vkUnmapMemory(device_, light_ubo_mem_); light_ubo_mapped_ = nullptr; }
    if (light_ubo_buf_)   { vkDestroyBuffer(device_, light_ubo_buf_, nullptr); light_ubo_buf_ = VK_NULL_HANDLE; }
    if (light_ubo_mem_)   { vkFreeMemory(device_, light_ubo_mem_, nullptr);    light_ubo_mem_ = VK_NULL_HANDLE; }

    // Mesh
    if (vertex_buf_) { vkDestroyBuffer(device_, vertex_buf_, nullptr); vertex_buf_ = VK_NULL_HANDLE; }
    if (vertex_mem_) { vkFreeMemory(device_, vertex_mem_, nullptr);    vertex_mem_ = VK_NULL_HANDLE; }
    if (index_buf_)  { vkDestroyBuffer(device_, index_buf_, nullptr);  index_buf_ = VK_NULL_HANDLE; }
    if (index_mem_)  { vkFreeMemory(device_, index_mem_, nullptr);     index_mem_ = VK_NULL_HANDLE; }

    cleanup_offscreen();

    initialized_ = false;
}

// ======================================================================
//  Mouse interaction
// ======================================================================
void SceneRenderer::process_input(float mouse_dx, float mouse_dy, float scroll,
                                  bool left_btn, bool right_btn, bool middle_btn) {
    const float orbit_speed = 0.01f;
    const float pan_speed   = 0.005f;
    const float zoom_speed  = 0.15f;

    if (left_btn) {
        camera.azimuth   -= mouse_dx * orbit_speed;
        camera.elevation += mouse_dy * orbit_speed;
        // Clamp elevation to avoid singularity
        camera.elevation = std::clamp(camera.elevation, -1.5f, 1.5f);
    }

    if (right_btn || middle_btn) {
        // Pan: move target in screen-aligned plane
        float ca = std::cos(camera.azimuth), sa = std::sin(camera.azimuth);
        vec3 right_dir = {ca, 0, -sa};
        vec3 up_dir    = {0, 1, 0};
        camera.target = camera.target +
            right_dir * (-mouse_dx * pan_speed * camera.distance) +
            up_dir    * ( mouse_dy * pan_speed * camera.distance);
    }

    if (scroll != 0.0f) {
        camera.distance *= (1.0f - scroll * zoom_speed);
        camera.distance = std::max(camera.distance, 0.1f);
    }
}

// ======================================================================
//  Prepare (called during ImGui frame — deferred GPU work)
// ======================================================================
void SceneRenderer::prepare(TensorTexture& texture) {
    if (!initialized_) return;

    // Lazy pipeline + descriptor set creation
    if (!pipeline_) {
        create_pipeline();
    }
    // (Re)create descriptor set if texture changed
    create_descriptor_set(texture);

    // Update UBOs
    update_ubos();

    needs_render_ = true;
}

// ======================================================================
//  Record render (called by Engine::end_frame into the main cmd buffer)
// ======================================================================
void SceneRenderer::record_render(VkCommandBuffer cmd) {
    if (!needs_render_) return;
    needs_render_ = false;

    // ── Begin offscreen render pass ───────────────────────────────
    std::array<VkClearValue, 3> clears{};
    clears[0].color = {{background[0], background[1], background[2], 1.0f}};
    clears[1].color = {{0, 0, 0, 1.0f}};                    // resolve (cleared by MSAA resolve)
    clears[2].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rp{};
    rp.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass        = render_pass_;
    rp.framebuffer       = framebuffer_;
    rp.renderArea.extent = {width_, height_};
    rp.clearValueCount   = static_cast<uint32_t>(clears.size());
    rp.pClearValues      = clears.data();

    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    // Bind pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

    // Viewport & scissor
    VkViewport viewport{0, 0, (float)width_, (float)height_, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{{0, 0}, {width_, height_}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Bind descriptor set
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout_, 0, 1, &scene_desc_, 0, nullptr);

    // Draw plane
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buf_, &offset);
    vkCmdBindIndexBuffer(cmd, index_buf_, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(cmd, index_count_, 1, 0, 0, 0);

    vkCmdEndRenderPass(cmd);
}

uintptr_t SceneRenderer::imgui_texture_id() const {
    return reinterpret_cast<uintptr_t>(imgui_desc_);
}

// ======================================================================
//  Resize / MSAA
// ======================================================================
void SceneRenderer::resize(uint32_t width, uint32_t height) {
    if (width == width_ && height == height_) return;
    vkDeviceWaitIdle(device_);
    width_  = width;
    height_ = height;

    if (imgui_desc_) { ImGui_ImplVulkan_RemoveTexture(imgui_desc_); imgui_desc_ = VK_NULL_HANDLE; }
    cleanup_offscreen();
    create_offscreen_resources();
    create_render_pass();
    create_framebuffer();

    // Force pipeline recreation (MSAA might differ)
    if (pipeline_) { vkDestroyPipeline(device_, pipeline_, nullptr); pipeline_ = VK_NULL_HANDLE; }
    if (pipeline_layout_) { vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr); pipeline_layout_ = VK_NULL_HANDLE; }
    if (desc_layout_) { vkDestroyDescriptorSetLayout(device_, desc_layout_, nullptr); desc_layout_ = VK_NULL_HANDLE; }
    if (scene_pool_) { vkDestroyDescriptorPool(device_, scene_pool_, nullptr); scene_pool_ = VK_NULL_HANDLE; }
    scene_desc_ = VK_NULL_HANDLE;
}

void SceneRenderer::set_msaa(VkSampleCountFlagBits msaa) {
    if (msaa == msaa_samples_) return;
    msaa_samples_ = msaa;

    // Rebuild offscreen resources with new sample count.
    // Cannot reuse resize() because it early-returns when dimensions are unchanged.
    vkDeviceWaitIdle(device_);

    if (imgui_desc_) { ImGui_ImplVulkan_RemoveTexture(imgui_desc_); imgui_desc_ = VK_NULL_HANDLE; }
    cleanup_offscreen();
    create_offscreen_resources();
    create_render_pass();
    create_framebuffer();

    // Force pipeline recreation (sample count changed)
    if (pipeline_)        { vkDestroyPipeline(device_, pipeline_, nullptr);             pipeline_ = VK_NULL_HANDLE; }
    if (pipeline_layout_) { vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr); pipeline_layout_ = VK_NULL_HANDLE; }
    if (desc_layout_)     { vkDestroyDescriptorSetLayout(device_, desc_layout_, nullptr); desc_layout_ = VK_NULL_HANDLE; }
    if (scene_pool_)      { vkDestroyDescriptorPool(device_, scene_pool_, nullptr);       scene_pool_ = VK_NULL_HANDLE; }
    scene_desc_ = VK_NULL_HANDLE;
}

// ======================================================================
//  Offscreen resources
// ======================================================================
void SceneRenderer::create_offscreen_resources() {
    VkFormat color_fmt = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat depth_fmt = VK_FORMAT_D32_SFLOAT;

    // MSAA color
    create_image(width_, height_, color_fmt, msaa_samples_,
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
                 msaa_color_, msaa_color_mem_);
    create_image_view(msaa_color_, color_fmt, VK_IMAGE_ASPECT_COLOR_BIT, msaa_color_view_);

    // Resolve target (1x) — also sampled by ImGui
    create_image(width_, height_, color_fmt, VK_SAMPLE_COUNT_1_BIT,
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                 resolve_color_, resolve_color_mem_);
    create_image_view(resolve_color_, color_fmt, VK_IMAGE_ASPECT_COLOR_BIT, resolve_view_);

    // MSAA depth
    create_image(width_, height_, depth_fmt, msaa_samples_,
                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
                 msaa_depth_, msaa_depth_mem_);
    create_image_view(msaa_depth_, depth_fmt, VK_IMAGE_ASPECT_DEPTH_BIT, msaa_depth_view_);

    // Resolve sampler
    if (!resolve_sampler_) {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(device_, &si, nullptr, &resolve_sampler_));
    }

    // Register resolve target with ImGui
    imgui_desc_ = ImGui_ImplVulkan_AddTexture(
        resolve_sampler_, resolve_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void SceneRenderer::cleanup_offscreen() {
    if (framebuffer_)      { vkDestroyFramebuffer(device_, framebuffer_, nullptr);    framebuffer_ = VK_NULL_HANDLE; }
    if (render_pass_)      { vkDestroyRenderPass(device_, render_pass_, nullptr);     render_pass_ = VK_NULL_HANDLE; }

    if (msaa_color_view_)  { vkDestroyImageView(device_, msaa_color_view_, nullptr);  msaa_color_view_ = VK_NULL_HANDLE; }
    if (msaa_color_)       { vkDestroyImage(device_, msaa_color_, nullptr);           msaa_color_ = VK_NULL_HANDLE; }
    if (msaa_color_mem_)   { vkFreeMemory(device_, msaa_color_mem_, nullptr);         msaa_color_mem_ = VK_NULL_HANDLE; }

    if (msaa_depth_view_)  { vkDestroyImageView(device_, msaa_depth_view_, nullptr);  msaa_depth_view_ = VK_NULL_HANDLE; }
    if (msaa_depth_)       { vkDestroyImage(device_, msaa_depth_, nullptr);           msaa_depth_ = VK_NULL_HANDLE; }
    if (msaa_depth_mem_)   { vkFreeMemory(device_, msaa_depth_mem_, nullptr);         msaa_depth_mem_ = VK_NULL_HANDLE; }

    if (resolve_view_)     { vkDestroyImageView(device_, resolve_view_, nullptr);     resolve_view_ = VK_NULL_HANDLE; }
    if (resolve_color_)    { vkDestroyImage(device_, resolve_color_, nullptr);        resolve_color_ = VK_NULL_HANDLE; }
    if (resolve_color_mem_) { vkFreeMemory(device_, resolve_color_mem_, nullptr);     resolve_color_mem_ = VK_NULL_HANDLE; }
}

// ======================================================================
//  Render pass (3 attachments: MSAA color, resolve, MSAA depth)
// ======================================================================
void SceneRenderer::create_render_pass() {
    VkAttachmentDescription attachments[3] = {};

    // [0] MSAA color
    attachments[0].format         = VK_FORMAT_R8G8B8A8_UNORM;
    attachments[0].samples        = msaa_samples_;
    attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE; // resolved
    attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // [1] Resolve target
    attachments[1].format         = VK_FORMAT_R8G8B8A8_UNORM;
    attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // [2] MSAA depth
    attachments[2].format         = VK_FORMAT_D32_SFLOAT;
    attachments[2].samples        = msaa_samples_;
    attachments[2].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[2].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[2].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[2].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[2].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference resolve_ref{1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depth_ref{2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &color_ref;
    subpass.pResolveAttachments     = &resolve_ref;
    subpass.pDepthStencilAttachment = &depth_ref;

    // Dependency: external → subpass 0
    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo ci{};
    ci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.attachmentCount = 3;
    ci.pAttachments    = attachments;
    ci.subpassCount    = 1;
    ci.pSubpasses      = &subpass;
    ci.dependencyCount = 1;
    ci.pDependencies   = &dep;

    VK_CHECK(vkCreateRenderPass(device_, &ci, nullptr, &render_pass_));
}

void SceneRenderer::create_framebuffer() {
    VkImageView views[] = {msaa_color_view_, resolve_view_, msaa_depth_view_};

    VkFramebufferCreateInfo ci{};
    ci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    ci.renderPass      = render_pass_;
    ci.attachmentCount = 3;
    ci.pAttachments    = views;
    ci.width           = width_;
    ci.height          = height_;
    ci.layers          = 1;
    VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &framebuffer_));
}

// ======================================================================
//  Pipeline
// ======================================================================
void SceneRenderer::create_pipeline() {
    // ── Descriptor set layout ─────────────────────────────────────
    VkDescriptorSetLayoutBinding bindings[3] = {};
    // binding 0: SceneUBO
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 1: LightUBO
    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    // binding 2: tensorTex (combined image sampler)
    bindings[2].binding         = 2;
    bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dsl_ci{};
    dsl_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = 3;
    dsl_ci.pBindings    = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &dsl_ci, nullptr, &desc_layout_));

    // ── Pipeline layout ───────────────────────────────────────────
    VkPipelineLayoutCreateInfo pl_ci{};
    pl_ci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 1;
    pl_ci.pSetLayouts    = &desc_layout_;
    VK_CHECK(vkCreatePipelineLayout(device_, &pl_ci, nullptr, &pipeline_layout_));

    // ── Shader modules ────────────────────────────────────────────
    VkShaderModule vert_mod = create_shader_module(plane_vert_spv, plane_vert_spv_size);
    VkShaderModule frag_mod = create_shader_module(plane_frag_spv, plane_frag_spv_size);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_mod;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag_mod;
    stages[1].pName  = "main";

    // ── Vertex input ──────────────────────────────────────────────
    VkVertexInputBindingDescription vtx_binding{};
    vtx_binding.binding   = 0;
    vtx_binding.stride    = sizeof(float) * 8; // pos(3) + normal(3) + uv(2)
    vtx_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vtx_attrs[3]{};
    vtx_attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};                          // position
    vtx_attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3};          // normal
    vtx_attrs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,    sizeof(float) * 6};          // uv

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &vtx_binding;
    vi.vertexAttributeDescriptionCount = 3;
    vi.pVertexAttributeDescriptions    = vtx_attrs;

    // ── Input assembly ────────────────────────────────────────────
    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // ── Viewport / scissor (dynamic) ──────────────────────────────
    VkPipelineViewportStateCreateInfo vp{};
    vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dyn_states;

    // ── Rasterizer ────────────────────────────────────────────────
    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = VK_CULL_MODE_NONE;  // show both sides of the plane
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth   = 1.0f;

    // ── Multisampling ─────────────────────────────────────────────
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = msaa_samples_;

    // ── Depth/stencil ─────────────────────────────────────────────
    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS;

    // ── Color blend ───────────────────────────────────────────────
    VkPipelineColorBlendAttachmentState blend_att{};
    blend_att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments    = &blend_att;

    // ── Create pipeline ───────────────────────────────────────────
    VkGraphicsPipelineCreateInfo gp{};
    gp.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gp.stageCount          = 2;
    gp.pStages             = stages;
    gp.pVertexInputState   = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState      = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState   = &ms;
    gp.pDepthStencilState  = &ds;
    gp.pColorBlendState    = &cb;
    gp.pDynamicState       = &dyn;
    gp.layout              = pipeline_layout_;
    gp.renderPass          = render_pass_;
    gp.subpass             = 0;

    VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gp, nullptr, &pipeline_));

    vkDestroyShaderModule(device_, vert_mod, nullptr);
    vkDestroyShaderModule(device_, frag_mod, nullptr);
}

// ======================================================================
//  Plane mesh
// ======================================================================
void SceneRenderer::create_plane_mesh() {
    // A simple quad in the XZ plane (Y-up), centered at origin
    float hw = plane.width  * 0.5f;
    float hh = plane.height * 0.5f;

    // Each vertex: pos(3) + normal(3) + uv(2)
    float vertices[] = {
        // pos              normal        uv
        -hw, 0.0f, -hh,   0, 1, 0,      0, 0,
         hw, 0.0f, -hh,   0, 1, 0,      1, 0,
         hw, 0.0f,  hh,   0, 1, 0,      1, 1,
        -hw, 0.0f,  hh,   0, 1, 0,      0, 1,
    };
    uint16_t indices[] = {0, 2, 1, 0, 3, 2};
    index_count_ = 6;

    // Create vertex buffer (host-visible for simplicity)
    auto create_buffer = [&](VkBuffer& buf, VkDeviceMemory& mem,
                             VkBufferUsageFlags usage, const void* data, VkDeviceSize size) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = size;
        bci.usage = usage;
        VK_CHECK(vkCreateBuffer(device_, &bci, nullptr, &buf));

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device_, buf, &req);
        VkMemoryAllocateInfo ai{};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &mem));
        VK_CHECK(vkBindBufferMemory(device_, buf, mem, 0));

        void* mapped;
        VK_CHECK(vkMapMemory(device_, mem, 0, size, 0, &mapped));
        std::memcpy(mapped, data, size);
        vkUnmapMemory(device_, mem);
    };

    create_buffer(vertex_buf_, vertex_mem_, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                  vertices, sizeof(vertices));
    create_buffer(index_buf_, index_mem_, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                  indices, sizeof(indices));
}

// ======================================================================
//  UBOs
// ======================================================================
void SceneRenderer::create_ubos() {
    auto create_ubo = [&](VkBuffer& buf, VkDeviceMemory& mem, void** mapped, VkDeviceSize size) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = size;
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        VK_CHECK(vkCreateBuffer(device_, &bci, nullptr, &buf));

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device_, buf, &req);
        VkMemoryAllocateInfo ai{};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &mem));
        VK_CHECK(vkBindBufferMemory(device_, buf, mem, 0));
        VK_CHECK(vkMapMemory(device_, mem, 0, size, 0, mapped));
    };

    create_ubo(ubo_buf_, ubo_mem_, &ubo_mapped_, sizeof(SceneUBO));
    create_ubo(light_ubo_buf_, light_ubo_mem_, &light_ubo_mapped_, sizeof(LightUBO));
}

void SceneRenderer::update_ubos() {
    float aspect = (height_ > 0) ? (float)width_ / (float)height_ : 1.0f;

    mat4 model = mat4::identity();
    mat4 view  = camera.view_matrix();
    mat4 proj  = camera.proj_matrix(aspect);
    vec3 eye   = camera.eye_position();

    SceneUBO scene;
    std::memcpy(scene.model, model.m, sizeof(float) * 16);
    std::memcpy(scene.view,  view.m,  sizeof(float) * 16);
    std::memcpy(scene.proj,  proj.m,  sizeof(float) * 16);
    scene.cameraPos[0] = eye.x;
    scene.cameraPos[1] = eye.y;
    scene.cameraPos[2] = eye.z;
    scene._pad0 = 0;
    std::memcpy(ubo_mapped_, &scene, sizeof(SceneUBO));

    LightUBO lt;
    vec3 dir = light.direction.normalized();
    lt.direction[0] = dir.x;
    lt.direction[1] = dir.y;
    lt.direction[2] = dir.z;
    lt.intensity     = light.enabled ? light.intensity : 0.0f;
    lt.color[0]      = light.color.x;
    lt.color[1]      = light.color.y;
    lt.color[2]      = light.color.z;
    lt.ambient       = light.ambient;
    lt.specular      = light.enabled ? light.specular : 0.0f;
    lt.shininess     = light.shininess;
    lt._pad0 = lt._pad1 = 0;
    std::memcpy(light_ubo_mapped_, &lt, sizeof(LightUBO));
}

// ======================================================================
//  Descriptor set
// ======================================================================
void SceneRenderer::create_descriptor_set(TensorTexture& texture) {
    // Create pool + set if needed
    if (!scene_pool_) {
        VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         2},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        };
        VkDescriptorPoolCreateInfo pi{};
        pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pi.maxSets       = 1;
        pi.poolSizeCount = 2;
        pi.pPoolSizes    = sizes;
        VK_CHECK(vkCreateDescriptorPool(device_, &pi, nullptr, &scene_pool_));
    }

    if (!scene_desc_) {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = scene_pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &desc_layout_;
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &scene_desc_));
    }

    // Update descriptors
    VkDescriptorBufferInfo scene_ubo_info{ubo_buf_, 0, sizeof(SceneUBO)};
    VkDescriptorBufferInfo light_ubo_info{light_ubo_buf_, 0, sizeof(LightUBO)};
    VkDescriptorImageInfo  tex_info{};
    tex_info.sampler     = texture.sampler();
    tex_info.imageView   = texture.image_view();
    tex_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet writes[3]{};
    // SceneUBO
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = scene_desc_;
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo     = &scene_ubo_info;
    // LightUBO
    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = scene_desc_;
    writes[1].dstBinding      = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].pBufferInfo     = &light_ubo_info;
    // Texture
    writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet          = scene_desc_;
    writes[2].dstBinding      = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo      = &tex_info;

    vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);
}

// ======================================================================
//  Helpers
// ======================================================================
VkShaderModule SceneRenderer::create_shader_module(const uint32_t* code, size_t size) {
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = size;
    ci.pCode    = code;
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(device_, &ci, nullptr, &mod));
    return mod;
}

uint32_t SceneRenderer::find_memory_type(uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(phys_device_, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
        if ((filter & (1u << i)) && (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("SceneRenderer: failed to find suitable memory type");
}

void SceneRenderer::create_image(uint32_t w, uint32_t h, VkFormat format,
                                 VkSampleCountFlagBits samples, VkImageUsageFlags usage,
                                 VkImage& image, VkDeviceMemory& memory) {
    VkImageCreateInfo ci{};
    ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType     = VK_IMAGE_TYPE_2D;
    ci.format        = format;
    ci.extent        = {w, h, 1};
    ci.mipLevels     = 1;
    ci.arrayLayers   = 1;
    ci.samples       = samples;
    ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ci.usage         = usage;
    ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(device_, &ci, nullptr, &image));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device_, image, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &memory));
    VK_CHECK(vkBindImageMemory(device_, image, memory, 0));
}

void SceneRenderer::create_image_view(VkImage image, VkFormat format,
                                      VkImageAspectFlags aspect, VkImageView& view) {
    VkImageViewCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ci.image    = image;
    ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ci.format   = format;
    ci.subresourceRange.aspectMask = aspect;
    ci.subresourceRange.levelCount = 1;
    ci.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &view));
}

} // namespace vultorch
