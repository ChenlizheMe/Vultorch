#include "bind_common.h"
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "engine.h"
#include "vk_utils.h"
#include "log.h"

#include <array>
#include <string>

void bind_engine(py::module_& m) {
    py::class_<vultorch::Engine>(m, "Engine")
        .def(py::init<>())
        .def("init", &vultorch::Engine::init,
             py::arg("title")  = "Vultorch",
             py::arg("width")  = 1280,
             py::arg("height") = 720,
             py::arg("vsync")  = true)
        .def("destroy",     &vultorch::Engine::destroy)
        .def("poll",        &vultorch::Engine::poll)
        .def("begin_frame", &vultorch::Engine::begin_frame)
        .def("end_frame",   &vultorch::Engine::end_frame)

        // ── TensorTexture (always available) ───────────────────────────
        .def("upload_tensor_cpu", [](vultorch::Engine& self, const std::string& name,
                                      uintptr_t data_ptr,
                                      uint32_t width, uint32_t height,
                                      uint32_t channels) {
            self.tensor_texture(name).upload_cpu(
                reinterpret_cast<const void*>(data_ptr), width, height, channels);
        }, py::arg("name") = "tensor",
           py::arg("data_ptr"), py::arg("width"), py::arg("height"),
           py::arg("channels") = 4)

        .def("tensor_texture_id", [](vultorch::Engine& self, const std::string& name) -> uintptr_t {
            return self.tensor_texture(name).imgui_texture_id();
        }, py::arg("name") = "tensor")
        .def("tensor_width",  [](vultorch::Engine& self, const std::string& name) {
            return self.tensor_texture(name).width();
        }, py::arg("name") = "tensor")
        .def("tensor_height", [](vultorch::Engine& self, const std::string& name) {
            return self.tensor_texture(name).height();
        }, py::arg("name") = "tensor")

        .def("set_tensor_filter", [](vultorch::Engine& self, const std::string& name, int mode) {
            self.tensor_texture(name).set_filter(
                mode == 0 ? vultorch::FilterMode::Nearest : vultorch::FilterMode::Linear);
        }, py::arg("name") = "tensor", py::arg("mode"))

#ifdef VULTORCH_HAS_CUDA
        // ── CUDA-specific tensor operations (zero-copy GPU interop) ────
        .def("allocate_shared_tensor", [](vultorch::Engine& self,
                                          const std::string& name,
                                          uint32_t width, uint32_t height,
                                          uint32_t channels, int device_id) -> uintptr_t {
            return self.tensor_texture(name).allocate_shared(width, height, channels, device_id);
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

        .def("is_shared_ptr", [](vultorch::Engine& self, const std::string& name, uintptr_t ptr) {
            return self.tensor_texture(name).is_shared_ptr(ptr);
        }, py::arg("name") = "tensor", py::arg("data_ptr"))
#endif

        // ── SceneRenderer (always available) ───────────────────────────
        .def("init_scene", [](vultorch::Engine& self,
                              uint32_t width, uint32_t height, int msaa) {
            self.scene_renderer(width, height, vultorch::msaa_from_int(msaa));
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
            self.wait_gpu();  // ensure no inflight frames reference old resources
            self.scene_renderer().resize(w, h);
        }, py::arg("width"), py::arg("height"))

        .def("scene_set_msaa", [](vultorch::Engine& self, int msaa) {
            self.wait_gpu();  // ensure no inflight frames reference old resources
            self.scene_renderer().set_msaa(vultorch::msaa_from_int(msaa));
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
    ;

    // ── Module-level log control ───────────────────────────────────
    m.def("set_log_level", [](const std::string& level) {
        if (level == "quiet" || level == "off")
            vultorch::log_level() = vultorch::LogLevel::Quiet;
        else if (level == "error")
            vultorch::log_level() = vultorch::LogLevel::Error;
        else if (level == "warn" || level == "warning")
            vultorch::log_level() = vultorch::LogLevel::Warn;
        else if (level == "info")
            vultorch::log_level() = vultorch::LogLevel::Info;
        else if (level == "debug")
            vultorch::log_level() = vultorch::LogLevel::Debug;
        else
            throw std::runtime_error("Unknown log level: " + level +
                " (use quiet/error/warn/info/debug)");
    }, py::arg("level"),
       "Set vultorch log verbosity: 'quiet', 'error', 'warn', 'info', 'debug'.");
}
