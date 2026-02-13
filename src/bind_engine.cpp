#include "bind_common.h"
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "engine.h"

#include <array>
#include <string>

// ── Minimal DLPack ABI structs for torch.from_dlpack() interop ───────
// Layout-compatible with the official DLPack v0.8 specification.
namespace {
struct VtDLDevice   { int32_t device_type; int32_t device_id; };
struct VtDLDataType { uint8_t code; uint8_t bits; uint16_t lanes; };
struct VtDLTensor {
    void*        data;
    VtDLDevice   device;
    int32_t      ndim;
    VtDLDataType dtype;
    int64_t*     shape;
    int64_t*     strides;
    uint64_t     byte_offset;
};
struct VtDLManagedTensor {
    VtDLTensor dl_tensor;
    void*      manager_ctx;
    void     (*deleter)(VtDLManagedTensor*);
};
} // anonymous namespace

void bind_engine(py::module_& m) {
    py::class_<vultorch::Engine>(m, "Engine")
        .def(py::init<>())
        .def("init", &vultorch::Engine::init,
             py::arg("title")  = "Vultorch",
             py::arg("width")  = 1280,
             py::arg("height") = 720)
        .def("destroy",     &vultorch::Engine::destroy)
        .def("poll",        &vultorch::Engine::poll)
        .def("begin_frame", &vultorch::Engine::begin_frame)
        .def("end_frame",   &vultorch::Engine::end_frame)
#ifdef VULTORCH_HAS_CUDA
        // ── TensorTexture (zero-copy GPU interop) ──────────────────────
        .def("allocate_shared_tensor", [](vultorch::Engine& self,
                                          const std::string& name,
                                          uint32_t width, uint32_t height,
                                          uint32_t channels, int device_id) -> uintptr_t {
            return self.tensor_texture(name).allocate_shared(width, height, channels, device_id);
        }, py::arg("name") = "tensor",
           py::arg("width"), py::arg("height"),
           py::arg("channels") = 4, py::arg("device_id") = 0)

        // DLPack capsule for torch.from_dlpack() — true zero-copy shared tensor
        .def("_allocate_shared_dlpack", [](vultorch::Engine& self,
                                           const std::string& name,
                                           uint32_t width, uint32_t height,
                                           uint32_t channels, int device_id) -> py::object {
            uintptr_t ptr = self.tensor_texture(name).allocate_shared(
                width, height, channels, device_id);
            if (ptr == 0)
                throw std::runtime_error("Failed to allocate shared GPU memory");

            auto* shape   = new int64_t[3]{(int64_t)height, (int64_t)width, (int64_t)channels};
            auto* managed = new VtDLManagedTensor{};
            managed->dl_tensor.data        = reinterpret_cast<void*>(ptr);
            managed->dl_tensor.device      = {2 /*kDLCUDA*/, device_id};
            managed->dl_tensor.ndim        = 3;
            managed->dl_tensor.dtype       = {2 /*kDLFloat*/, 32, 1};
            managed->dl_tensor.shape       = shape;
            managed->dl_tensor.strides     = nullptr;  // C-contiguous
            managed->dl_tensor.byte_offset = 0;
            managed->manager_ctx = nullptr;
            managed->deleter = [](VtDLManagedTensor* self) {
                delete[] self->dl_tensor.shape;
                delete self;
            };

            PyObject* capsule = PyCapsule_New(
                static_cast<void*>(managed), "dltensor",
                [](PyObject* cap) {
                    void* p = PyCapsule_GetPointer(cap, "dltensor");
                    if (p) {
                        auto* m = static_cast<VtDLManagedTensor*>(p);
                        if (m->deleter) m->deleter(m);
                    }
                });
            if (!capsule) throw py::error_already_set();
            return py::reinterpret_steal<py::object>(capsule);
        }, py::arg("name") = "tensor",
           py::arg("width"), py::arg("height"),
           py::arg("channels") = 4, py::arg("device_id") = 0)

        .def("upload_tensor", [](vultorch::Engine& self, const std::string& name,
                                  uintptr_t data_ptr,
                                  uint32_t width, uint32_t height,
                                  uint32_t channels, int device_id) {
            self.tensor_texture(name).upload(data_ptr, width, height, channels, device_id);
        }, py::arg("name") = "tensor",
           py::arg("data_ptr"), py::arg("width"), py::arg("height"),
           py::arg("channels") = 4, py::arg("device_id") = 0)

        .def("sync_tensor", [](vultorch::Engine& self, const std::string& name) {
            self.tensor_texture(name).sync();
        }, py::arg("name") = "tensor")

        .def("tensor_texture_id", [](vultorch::Engine& self, const std::string& name) -> uintptr_t {
            return self.tensor_texture(name).imgui_texture_id();
        }, py::arg("name") = "tensor")
        .def("tensor_width",  [](vultorch::Engine& self, const std::string& name) {
            return self.tensor_texture(name).width();
        }, py::arg("name") = "tensor")
        .def("tensor_height", [](vultorch::Engine& self, const std::string& name) {
            return self.tensor_texture(name).height();
        }, py::arg("name") = "tensor")

        .def("is_shared_ptr", [](vultorch::Engine& self, const std::string& name, uintptr_t ptr) {
            return self.tensor_texture(name).is_shared_ptr(ptr);
        }, py::arg("name") = "tensor", py::arg("data_ptr"))

        .def("set_tensor_filter", [](vultorch::Engine& self, const std::string& name, int mode) {
            self.tensor_texture(name).set_filter(
                mode == 0 ? vultorch::FilterMode::Nearest : vultorch::FilterMode::Linear);
        }, py::arg("name") = "tensor", py::arg("mode"))

        // ── SceneRenderer ──────────────────────────────────────────────
        .def("init_scene", [](vultorch::Engine& self,
                              uint32_t width, uint32_t height, int msaa) {
            VkSampleCountFlagBits samples;
            switch (msaa) {
                case 1: samples = VK_SAMPLE_COUNT_1_BIT; break;
                case 2: samples = VK_SAMPLE_COUNT_2_BIT; break;
                case 8: samples = VK_SAMPLE_COUNT_8_BIT; break;
                default: samples = VK_SAMPLE_COUNT_4_BIT; break;
            }
            self.scene_renderer(width, height, samples);
        }, py::arg("width") = 800, py::arg("height") = 600, py::arg("msaa") = 4)

        .def("scene_render", [](vultorch::Engine& self, const std::string& texture_name) {
            auto& sr = self.scene_renderer();
            auto& tt = self.tensor_texture(texture_name);
            sr.prepare(tt);
        }, py::arg("texture_name") = "scene")

        .def("scene_texture_id", [](vultorch::Engine& self) -> uintptr_t {
            return self.scene_renderer().imgui_texture_id();
        })

        .def("scene_resize", [](vultorch::Engine& self, uint32_t w, uint32_t h) {
            self.scene_renderer().resize(w, h);
        }, py::arg("width"), py::arg("height"))

        .def("scene_set_msaa", [](vultorch::Engine& self, int msaa) {
            VkSampleCountFlagBits samples;
            switch (msaa) {
                case 1: samples = VK_SAMPLE_COUNT_1_BIT; break;
                case 2: samples = VK_SAMPLE_COUNT_2_BIT; break;
                case 8: samples = VK_SAMPLE_COUNT_8_BIT; break;
                default: samples = VK_SAMPLE_COUNT_4_BIT; break;
            }
            self.scene_renderer().set_msaa(samples);
        }, py::arg("msaa"))

        .def("scene_process_input", [](vultorch::Engine& self,
                                       float dx, float dy, float scroll,
                                       bool left, bool right, bool middle) {
            self.scene_renderer().process_input(dx, dy, scroll, left, right, middle);
        }, py::arg("dx"), py::arg("dy"), py::arg("scroll"),
           py::arg("left"), py::arg("right"), py::arg("middle"))

        .def("scene_set_camera", [](vultorch::Engine& self, float azimuth, float elevation,
                                    float distance, float tx, float ty, float tz, float fov) {
            auto& cam = self.scene_renderer().camera;
            cam.azimuth   = azimuth;
            cam.elevation = elevation;
            cam.distance  = distance;
            cam.target    = {tx, ty, tz};
            cam.fov       = fov;
        }, py::arg("azimuth"), py::arg("elevation"), py::arg("distance"),
           py::arg("tx") = 0.f, py::arg("ty") = 0.f, py::arg("tz") = 0.f,
           py::arg("fov") = 45.f)

        .def("scene_get_camera", [](vultorch::Engine& self) {
            auto& cam = self.scene_renderer().camera;
            return py::make_tuple(cam.azimuth, cam.elevation, cam.distance,
                                  cam.target.x, cam.target.y, cam.target.z, cam.fov);
        })

        .def("scene_set_light", [](vultorch::Engine& self,
                                   float dx, float dy, float dz,
                                   float intensity, float r, float g, float b,
                                   float ambient, float specular, float shininess,
                                   bool enabled) {
            auto& lt = self.scene_renderer().light;
            lt.direction = {dx, dy, dz};
            lt.intensity = intensity;
            lt.color     = {r, g, b};
            lt.ambient   = ambient;
            lt.specular  = specular;
            lt.shininess = shininess;
            lt.enabled   = enabled;
        }, py::arg("dx") = 0.3f, py::arg("dy") = -1.f, py::arg("dz") = 0.5f,
           py::arg("intensity") = 1.f,
           py::arg("r") = 1.f, py::arg("g") = 1.f, py::arg("b") = 1.f,
           py::arg("ambient") = 0.15f, py::arg("specular") = 0.5f,
           py::arg("shininess") = 32.f, py::arg("enabled") = true)

        .def("scene_set_background", [](vultorch::Engine& self, float r, float g, float b) {
            auto& bg = self.scene_renderer().background;
            bg[0] = r; bg[1] = g; bg[2] = b;
        }, py::arg("r") = 0.12f, py::arg("g") = 0.12f, py::arg("b") = 0.14f)

        .def("scene_width", [](vultorch::Engine& self) { return self.scene_renderer().width(); })
        .def("scene_height", [](vultorch::Engine& self) { return self.scene_renderer().height(); })

        .def("max_msaa", [](vultorch::Engine& self) -> int {
            return static_cast<int>(self.max_msaa_samples());
        })
#endif
    ;
}
