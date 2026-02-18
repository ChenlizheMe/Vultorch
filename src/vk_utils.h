#pragma once
/// vk_utils.h — Shared Vulkan utilities (error checking, memory helpers, MSAA).
///
/// Single-header utility: all functions are inline to avoid linker issues
/// across multiple translation units.

#include <vulkan/vulkan.h>
#include <stdexcept>
#include <string>
#include <cstdint>

namespace vultorch {

// ── Human-readable Vulkan result codes ────────────────────────────────
inline const char* vk_result_string(VkResult r) {
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

// ── find_memory_type (shared by TensorTexture & SceneRenderer) ────────
inline uint32_t find_memory_type(VkPhysicalDevice phys_device,
                                 uint32_t filter,
                                 VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(phys_device, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
        if ((filter & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("[vultorch] Failed to find suitable memory type");
}

// ── MSAA int → VkSampleCountFlagBits ──────────────────────────────────
inline VkSampleCountFlagBits msaa_from_int(int msaa) {
    switch (msaa) {
        case 1:  return VK_SAMPLE_COUNT_1_BIT;
        case 2:  return VK_SAMPLE_COUNT_2_BIT;
        case 8:  return VK_SAMPLE_COUNT_8_BIT;
        case 16: return VK_SAMPLE_COUNT_16_BIT;
        case 32: return VK_SAMPLE_COUNT_32_BIT;
        case 64: return VK_SAMPLE_COUNT_64_BIT;
        default: return VK_SAMPLE_COUNT_4_BIT;
    }
}

} // namespace vultorch

// ── Unified VK_CHECK macro ────────────────────────────────────────────
// Human-readable error name + numeric code + file:line.
#define VK_CHECK(x)                                                             \
    do {                                                                        \
        VkResult _r = (x);                                                      \
        if (_r != VK_SUCCESS)                                                   \
            throw std::runtime_error(                                           \
                std::string("[vultorch] Vulkan error: ") +                      \
                vultorch::vk_result_string(_r) + " (" +                        \
                std::to_string(static_cast<int>(_r)) + ") at " +               \
                __FILE__ + ":" + std::to_string(__LINE__));                     \
    } while (0)

// ── CUDA error-check macro (only when CUDA is available) ──────────────
#ifdef VULTORCH_HAS_CUDA
#include <cuda.h>
#define CU_CHECK(x)                                                             \
    do {                                                                        \
        CUresult _r = (x);                                                      \
        if (_r != CUDA_SUCCESS) {                                               \
            const char* msg = nullptr;                                          \
            cuGetErrorString(_r, &msg);                                         \
            throw std::runtime_error(                                           \
                std::string("[vultorch] CUDA error: ") +                        \
                (msg ? msg : "unknown") +                                       \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));            \
        }                                                                       \
    } while (0)
#endif // VULTORCH_HAS_CUDA
