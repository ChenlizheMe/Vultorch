// ── Vultorch image I/O bindings (stb_image) ───────────────────────────
// Provides _imread / _imwrite at the C++ level.
// Python wrappers in vultorch/__init__.py handle tensor creation / device.

#include "bind_common.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// stb implementations — defined exactly once in this TU
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────
// _imread(path, desired_channels) → (bytes, width, height, channels)
//
// Returns raw float32 pixel data as a Python bytes object together with
// the image dimensions.  The Python side wraps this into a torch.Tensor
// and moves it to the desired device.
// ─────────────────────────────────────────────────────────────────────
static py::tuple imread_impl(const std::string& path, int desired_channels)
{
    int w = 0, h = 0, file_channels = 0;

    // stbi_loadf returns float32 data in [0, 1]
    float* data = stbi_loadf(path.c_str(), &w, &h, &file_channels,
                             desired_channels);
    if (!data) {
        throw std::runtime_error(
            "Failed to load image '" + path + "': " +
            std::string(stbi_failure_reason()));
    }

    int out_channels = desired_channels > 0 ? desired_channels : file_channels;
    size_t nbytes = static_cast<size_t>(w) * h * out_channels * sizeof(float);

    // Copy into a Python bytes object and free the stb buffer
    py::bytes result(reinterpret_cast<const char*>(data), nbytes);
    stbi_image_free(data);

    return py::make_tuple(result, w, h, out_channels);
}

// ─────────────────────────────────────────────────────────────────────
// _imwrite(path, data_ptr, width, height, channels, quality)
//
// Writes a float32 RGBA/RGB/grey buffer to disk.
// Format is inferred from file extension (.png, .jpg, .bmp, .tga, .hdr).
// float32 → uint8 conversion is done here for LDR formats.
// ─────────────────────────────────────────────────────────────────────
static void imwrite_impl(const std::string& path,
                         uintptr_t data_ptr,
                         int w, int h, int channels,
                         int quality)
{
    const float* fdata = reinterpret_cast<const float*>(data_ptr);

    // Determine format from extension
    auto ext_pos = path.rfind('.');
    std::string ext;
    if (ext_pos != std::string::npos) {
        ext = path.substr(ext_pos);
        for (auto& c : ext) c = static_cast<char>(std::tolower(c));
    }

    int ok = 0;

    if (ext == ".hdr") {
        // HDR: write float32 directly
        ok = stbi_write_hdr(path.c_str(), w, h, channels, fdata);
    } else {
        // LDR formats: convert float32 [0,1] → uint8 [0,255]
        size_t n = static_cast<size_t>(w) * h * channels;
        std::vector<uint8_t> u8(n);
        for (size_t i = 0; i < n; ++i) {
            float v = fdata[i];
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            u8[i] = static_cast<uint8_t>(v * 255.0f + 0.5f);
        }

        if (ext == ".png") {
            ok = stbi_write_png(path.c_str(), w, h, channels,
                                u8.data(), w * channels);
        } else if (ext == ".jpg" || ext == ".jpeg") {
            ok = stbi_write_jpg(path.c_str(), w, h, channels,
                                u8.data(), quality);
        } else if (ext == ".bmp") {
            ok = stbi_write_bmp(path.c_str(), w, h, channels, u8.data());
        } else if (ext == ".tga") {
            ok = stbi_write_tga(path.c_str(), w, h, channels, u8.data());
        } else {
            throw std::runtime_error(
                "Unsupported image format '" + ext +
                "'. Supported: .png, .jpg, .bmp, .tga, .hdr");
        }
    }

    if (!ok) {
        throw std::runtime_error("Failed to write image '" + path + "'");
    }
}

// ─────────────────────────────────────────────────────────────────────
// pybind11 registration
// ─────────────────────────────────────────────────────────────────────
void bind_io(py::module_& m)
{
    m.def("_imread", &imread_impl,
          py::arg("path"), py::arg("desired_channels") = 4,
          "Load an image from disk as raw float32 bytes.\n"
          "Returns (bytes, width, height, channels).");

    m.def("_imwrite", &imwrite_impl,
          py::arg("path"), py::arg("data_ptr"),
          py::arg("width"), py::arg("height"),
          py::arg("channels"), py::arg("quality") = 95,
          "Write a float32 tensor buffer to an image file on disk.");
}
