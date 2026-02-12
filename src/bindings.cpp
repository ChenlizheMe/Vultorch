#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "engine.h"
#include <imgui_internal.h>   // DockBuilder APIs

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

namespace py = pybind11;

PYBIND11_MODULE(_vultorch, m) {
    m.doc() = "Vultorch – Vulkan + PyTorch zero-copy interop with ImGui";

    // ------------------------------------------------------------------ Engine
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

            // PyCapsule named "dltensor" — consumed by torch.from_dlpack()
            PyObject* capsule = PyCapsule_New(
                static_cast<void*>(managed), "dltensor",
                [](PyObject* cap) {
                    // Only called if torch.from_dlpack() did NOT consume it
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
        }, py::arg("name") = "tensor", py::arg("mode"))  // 0=nearest, 1=linear

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

    m.attr("HAS_CUDA") = 
#ifdef VULTORCH_HAS_CUDA
        true;
#else
        false;
#endif

    // ==================================================================
    //  ImGui Python bindings — comprehensive set
    // ==================================================================
    auto ig = m.def_submodule("ui", "Dear ImGui bindings for Python");

    // ---------- Windows ------------------------------------------------
    ig.def("begin", [](const char* name, bool opened, int flags) {
        bool o = opened;
        bool visible = ImGui::Begin(name, opened ? &o : nullptr, flags);
        return py::make_tuple(visible, o);
    }, py::arg("name"), py::arg("opened") = true, py::arg("flags") = 0);

    ig.def("end", []() { ImGui::End(); });

    ig.def("begin_child", [](const char* id, float w, float h, int child_flags, int window_flags) {
        return ImGui::BeginChild(id, ImVec2(w, h), child_flags, window_flags);
    }, py::arg("id"), py::arg("width") = 0.f, py::arg("height") = 0.f,
       py::arg("child_flags") = 0, py::arg("window_flags") = 0);
    ig.def("end_child", []() { ImGui::EndChild(); });

    // ---------- Text / Labels ------------------------------------------
    ig.def("text", [](const char* t) { ImGui::TextUnformatted(t); }, py::arg("text"));
    ig.def("text_colored", [](float r, float g, float b, float a, const char* t) {
        ImGui::TextColored(ImVec4(r, g, b, a), "%s", t);
    }, py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a"), py::arg("text"));
    ig.def("text_disabled", [](const char* t) { ImGui::TextDisabled("%s", t); }, py::arg("text"));
    ig.def("text_wrapped", [](const char* t) { ImGui::TextWrapped("%s", t); }, py::arg("text"));
    ig.def("label_text", [](const char* label, const char* t) {
        ImGui::LabelText(label, "%s", t);
    }, py::arg("label"), py::arg("text"));
    ig.def("bullet_text", [](const char* t) { ImGui::BulletText("%s", t); }, py::arg("text"));

    // ---------- Buttons ------------------------------------------------
    ig.def("button", [](const char* label, float w, float h) {
        return ImGui::Button(label, ImVec2(w, h));
    }, py::arg("label"), py::arg("width") = 0.f, py::arg("height") = 0.f);

    ig.def("small_button", [](const char* label) {
        return ImGui::SmallButton(label);
    }, py::arg("label"));

    ig.def("invisible_button", [](const char* id, float w, float h) {
        return ImGui::InvisibleButton(id, ImVec2(w, h));
    }, py::arg("id"), py::arg("width"), py::arg("height"));

    ig.def("arrow_button", [](const char* id, int dir) {
        return ImGui::ArrowButton(id, static_cast<ImGuiDir>(dir));
    }, py::arg("id"), py::arg("dir"));

    ig.def("radio_button", [](const char* label, bool active) {
        return ImGui::RadioButton(label, active);
    }, py::arg("label"), py::arg("active"));

    // ---------- Checkbox -----------------------------------------------
    ig.def("checkbox", [](const char* label, bool v) {
        ImGui::Checkbox(label, &v);
        return v;
    }, py::arg("label"), py::arg("value"));

    // ---------- Sliders ------------------------------------------------
    ig.def("slider_float", [](const char* label, float v, float lo, float hi, const char* fmt) {
        ImGui::SliderFloat(label, &v, lo, hi, fmt);
        return v;
    }, py::arg("label"), py::arg("value"), py::arg("min") = 0.f, py::arg("max") = 1.f,
       py::arg("format") = "%.3f");

    ig.def("slider_float2", [](const char* label, float v0, float v1, float lo, float hi) {
        float v[2] = {v0, v1};
        ImGui::SliderFloat2(label, v, lo, hi);
        return py::make_tuple(v[0], v[1]);
    }, py::arg("label"), py::arg("v0"), py::arg("v1"), py::arg("min") = 0.f, py::arg("max") = 1.f);

    ig.def("slider_float3", [](const char* label, float v0, float v1, float v2, float lo, float hi) {
        float v[3] = {v0, v1, v2};
        ImGui::SliderFloat3(label, v, lo, hi);
        return py::make_tuple(v[0], v[1], v[2]);
    }, py::arg("label"), py::arg("v0"), py::arg("v1"), py::arg("v2"),
       py::arg("min") = 0.f, py::arg("max") = 1.f);

    ig.def("slider_float4", [](const char* label, float v0, float v1, float v2, float v3, float lo, float hi) {
        float v[4] = {v0, v1, v2, v3};
        ImGui::SliderFloat4(label, v, lo, hi);
        return py::make_tuple(v[0], v[1], v[2], v[3]);
    }, py::arg("label"), py::arg("v0"), py::arg("v1"), py::arg("v2"), py::arg("v3"),
       py::arg("min") = 0.f, py::arg("max") = 1.f);

    ig.def("slider_int", [](const char* label, int v, int lo, int hi) {
        ImGui::SliderInt(label, &v, lo, hi);
        return v;
    }, py::arg("label"), py::arg("value"), py::arg("min") = 0, py::arg("max") = 100);

    ig.def("slider_angle", [](const char* label, float rad, float lo, float hi) {
        ImGui::SliderAngle(label, &rad, lo, hi);
        return rad;
    }, py::arg("label"), py::arg("value_rad"), py::arg("min_deg") = -360.f, py::arg("max_deg") = 360.f);

    // ---------- Drag inputs --------------------------------------------
    ig.def("drag_float", [](const char* label, float v, float speed, float lo, float hi) {
        ImGui::DragFloat(label, &v, speed, lo, hi);
        return v;
    }, py::arg("label"), py::arg("value"), py::arg("speed") = 1.f,
       py::arg("min") = 0.f, py::arg("max") = 0.f);

    ig.def("drag_float2", [](const char* label, float v0, float v1, float speed, float lo, float hi) {
        float v[2] = {v0, v1};
        ImGui::DragFloat2(label, v, speed, lo, hi);
        return py::make_tuple(v[0], v[1]);
    }, py::arg("label"), py::arg("v0"), py::arg("v1"), py::arg("speed") = 1.f,
       py::arg("min") = 0.f, py::arg("max") = 0.f);

    ig.def("drag_float3", [](const char* label, float v0, float v1, float v2, float speed, float lo, float hi) {
        float v[3] = {v0, v1, v2};
        ImGui::DragFloat3(label, v, speed, lo, hi);
        return py::make_tuple(v[0], v[1], v[2]);
    }, py::arg("label"), py::arg("v0"), py::arg("v1"), py::arg("v2"),
       py::arg("speed") = 1.f, py::arg("min") = 0.f, py::arg("max") = 0.f);

    ig.def("drag_int", [](const char* label, int v, float speed, int lo, int hi) {
        ImGui::DragInt(label, &v, speed, lo, hi);
        return v;
    }, py::arg("label"), py::arg("value"), py::arg("speed") = 1.f,
       py::arg("min") = 0, py::arg("max") = 0);

    // ---------- Input fields -------------------------------------------
    ig.def("input_float", [](const char* label, float v, float step, float step_fast) {
        ImGui::InputFloat(label, &v, step, step_fast);
        return v;
    }, py::arg("label"), py::arg("value"), py::arg("step") = 0.f, py::arg("step_fast") = 0.f);

    ig.def("input_float2", [](const char* label, float v0, float v1) {
        float v[2] = {v0, v1};
        ImGui::InputFloat2(label, v);
        return py::make_tuple(v[0], v[1]);
    }, py::arg("label"), py::arg("v0"), py::arg("v1"));

    ig.def("input_float3", [](const char* label, float v0, float v1, float v2) {
        float v[3] = {v0, v1, v2};
        ImGui::InputFloat3(label, v);
        return py::make_tuple(v[0], v[1], v[2]);
    }, py::arg("label"), py::arg("v0"), py::arg("v1"), py::arg("v2"));

    ig.def("input_float4", [](const char* label, float v0, float v1, float v2, float v3) {
        float v[4] = {v0, v1, v2, v3};
        ImGui::InputFloat4(label, v);
        return py::make_tuple(v[0], v[1], v[2], v[3]);
    }, py::arg("label"), py::arg("v0"), py::arg("v1"), py::arg("v2"), py::arg("v3"));

    ig.def("input_int", [](const char* label, int v, int step, int step_fast) {
        ImGui::InputInt(label, &v, step, step_fast);
        return v;
    }, py::arg("label"), py::arg("value"), py::arg("step") = 1, py::arg("step_fast") = 100);

    ig.def("input_text", [](const char* label, std::string text, int max_len, int flags) {
        text.resize(static_cast<size_t>(max_len));
        ImGui::InputText(label, text.data(), static_cast<size_t>(max_len), flags);
        return std::string(text.c_str());
    }, py::arg("label"), py::arg("text") = "", py::arg("max_length") = 256, py::arg("flags") = 0);

    ig.def("input_text_multiline", [](const char* label, std::string text, int max_len, float w, float h, int flags) {
        text.resize(static_cast<size_t>(max_len));
        ImGui::InputTextMultiline(label, text.data(), static_cast<size_t>(max_len), ImVec2(w, h), flags);
        return std::string(text.c_str());
    }, py::arg("label"), py::arg("text") = "", py::arg("max_length") = 1024,
       py::arg("width") = 0.f, py::arg("height") = 0.f, py::arg("flags") = 0);

    // ---------- Color --------------------------------------------------
    ig.def("color_edit3", [](const char* label, float r, float g, float b, int flags) {
        float c[3] = {r, g, b};
        ImGui::ColorEdit3(label, c, flags);
        return py::make_tuple(c[0], c[1], c[2]);
    }, py::arg("label"), py::arg("r") = 0.f, py::arg("g") = 0.f, py::arg("b") = 0.f, py::arg("flags") = 0);

    ig.def("color_edit4", [](const char* label, float r, float g, float b, float a, int flags) {
        float c[4] = {r, g, b, a};
        ImGui::ColorEdit4(label, c, flags);
        return py::make_tuple(c[0], c[1], c[2], c[3]);
    }, py::arg("label"), py::arg("r") = 0.f, py::arg("g") = 0.f, py::arg("b") = 0.f,
       py::arg("a") = 1.f, py::arg("flags") = 0);

    ig.def("color_picker3", [](const char* label, float r, float g, float b, int flags) {
        float c[3] = {r, g, b};
        ImGui::ColorPicker3(label, c, flags);
        return py::make_tuple(c[0], c[1], c[2]);
    }, py::arg("label"), py::arg("r") = 0.f, py::arg("g") = 0.f, py::arg("b") = 0.f, py::arg("flags") = 0);

    ig.def("color_picker4", [](const char* label, float r, float g, float b, float a, int flags) {
        float c[4] = {r, g, b, a};
        ImGui::ColorPicker4(label, c, flags);
        return py::make_tuple(c[0], c[1], c[2], c[3]);
    }, py::arg("label"), py::arg("r") = 0.f, py::arg("g") = 0.f, py::arg("b") = 0.f,
       py::arg("a") = 1.f, py::arg("flags") = 0);

    // ---------- Combo / Listbox ----------------------------------------
    ig.def("combo", [](const char* label, int current, std::vector<std::string> items) {
        // Build null-separated string for ImGui
        std::vector<const char*> ptrs;
        ptrs.reserve(items.size());
        for (auto& s : items) ptrs.push_back(s.c_str());
        ImGui::Combo(label, &current, ptrs.data(), static_cast<int>(ptrs.size()));
        return current;
    }, py::arg("label"), py::arg("current"), py::arg("items"));

    ig.def("listbox", [](const char* label, int current, std::vector<std::string> items, int height_items) {
        std::vector<const char*> ptrs;
        ptrs.reserve(items.size());
        for (auto& s : items) ptrs.push_back(s.c_str());
        ImGui::ListBox(label, &current, ptrs.data(), static_cast<int>(ptrs.size()), height_items);
        return current;
    }, py::arg("label"), py::arg("current"), py::arg("items"), py::arg("height_items") = -1);

    // ---------- Tree / Collapsing headers ------------------------------
    ig.def("tree_node", [](const char* label) { return ImGui::TreeNode(label); }, py::arg("label"));
    ig.def("tree_pop", []() { ImGui::TreePop(); });

    ig.def("collapsing_header", [](const char* label, int flags) {
        return ImGui::CollapsingHeader(label, flags);
    }, py::arg("label"), py::arg("flags") = 0);

    // ---------- Selectable ---------------------------------------------
    ig.def("selectable", [](const char* label, bool selected, int flags, float w, float h) {
        ImGui::Selectable(label, &selected, flags, ImVec2(w, h));
        return selected;
    }, py::arg("label"), py::arg("selected") = false, py::arg("flags") = 0,
       py::arg("width") = 0.f, py::arg("height") = 0.f);

    // ---------- Tabs ---------------------------------------------------
    ig.def("begin_tab_bar", [](const char* id, int flags) {
        return ImGui::BeginTabBar(id, flags);
    }, py::arg("id"), py::arg("flags") = 0);
    ig.def("end_tab_bar", []() { ImGui::EndTabBar(); });

    ig.def("begin_tab_item", [](const char* label) {
        return ImGui::BeginTabItem(label);
    }, py::arg("label"));
    ig.def("end_tab_item", []() { ImGui::EndTabItem(); });

    // ---------- Menu bar -----------------------------------------------
    ig.def("begin_main_menu_bar", []() { return ImGui::BeginMainMenuBar(); });
    ig.def("end_main_menu_bar",   []() { ImGui::EndMainMenuBar(); });
    ig.def("begin_menu_bar",      []() { return ImGui::BeginMenuBar(); });
    ig.def("end_menu_bar",        []() { ImGui::EndMenuBar(); });
    ig.def("begin_menu", [](const char* label, bool enabled) {
        return ImGui::BeginMenu(label, enabled);
    }, py::arg("label"), py::arg("enabled") = true);
    ig.def("end_menu", []() { ImGui::EndMenu(); });
    ig.def("menu_item", [](const char* label, const char* shortcut, bool selected, bool enabled) {
        ImGui::MenuItem(label, shortcut, &selected, enabled);
        return selected;
    }, py::arg("label"), py::arg("shortcut") = "",
       py::arg("selected") = false, py::arg("enabled") = true);

    // ---------- Popups / Modals ----------------------------------------
    ig.def("open_popup", [](const char* id) { ImGui::OpenPopup(id); }, py::arg("id"));
    ig.def("begin_popup", [](const char* id, int flags) {
        return ImGui::BeginPopup(id, flags);
    }, py::arg("id"), py::arg("flags") = 0);
    ig.def("begin_popup_modal", [](const char* name, int flags) {
        return ImGui::BeginPopupModal(name, nullptr, flags);
    }, py::arg("name"), py::arg("flags") = 0);
    ig.def("end_popup", []() { ImGui::EndPopup(); });
    ig.def("close_current_popup", []() { ImGui::CloseCurrentPopup(); });

    // ---------- Tooltips -----------------------------------------------
    ig.def("begin_tooltip", []() { return ImGui::BeginTooltip(); });
    ig.def("end_tooltip",   []() { ImGui::EndTooltip(); });
    ig.def("set_tooltip", [](const char* t) { ImGui::SetTooltip("%s", t); }, py::arg("text"));

    // ---------- Layout / Spacing ---------------------------------------
    ig.def("separator",     []() { ImGui::Separator(); });
    ig.def("same_line",     [](float offset, float spacing) { ImGui::SameLine(offset, spacing); },
           py::arg("offset") = 0.f, py::arg("spacing") = -1.f);
    ig.def("new_line",      []() { ImGui::NewLine(); });
    ig.def("spacing",       []() { ImGui::Spacing(); });
    ig.def("dummy",         [](float w, float h) { ImGui::Dummy(ImVec2(w, h)); },
           py::arg("width"), py::arg("height"));
    ig.def("indent",        [](float w) { ImGui::Indent(w); }, py::arg("width") = 0.f);
    ig.def("unindent",      [](float w) { ImGui::Unindent(w); }, py::arg("width") = 0.f);
    ig.def("begin_group",   []() { ImGui::BeginGroup(); });
    ig.def("end_group",     []() { ImGui::EndGroup(); });

    ig.def("push_item_width", [](float w) { ImGui::PushItemWidth(w); }, py::arg("width"));
    ig.def("pop_item_width",  []() { ImGui::PopItemWidth(); });

    ig.def("columns", [](int count, const char* id, bool border) {
        ImGui::Columns(count, id, border);
    }, py::arg("count") = 1, py::arg("id") = "", py::arg("border") = true);
    ig.def("next_column", []() { ImGui::NextColumn(); });

    // ---------- Tables -------------------------------------------------
    ig.def("begin_table", [](const char* id, int columns, int flags, float ow, float oh, float iw) {
        return ImGui::BeginTable(id, columns, flags, ImVec2(ow, oh), iw);
    }, py::arg("id"), py::arg("columns"), py::arg("flags") = 0,
       py::arg("outer_width") = 0.f, py::arg("outer_height") = 0.f, py::arg("inner_width") = 0.f);
    ig.def("end_table",          []() { ImGui::EndTable(); });
    ig.def("table_next_row",     [](int flags, float min_h) { ImGui::TableNextRow(flags, min_h); },
           py::arg("flags") = 0, py::arg("min_row_height") = 0.f);
    ig.def("table_next_column",  []() { return ImGui::TableNextColumn(); });
    ig.def("table_set_column_index", [](int idx) { return ImGui::TableSetColumnIndex(idx); },
           py::arg("index"));
    ig.def("table_setup_column", [](const char* label, int flags, float init_w) {
        ImGui::TableSetupColumn(label, flags, init_w);
    }, py::arg("label"), py::arg("flags") = 0, py::arg("init_width") = 0.f);
    ig.def("table_headers_row",  []() { ImGui::TableHeadersRow(); });

    // ---------- ID stack -----------------------------------------------
    ig.def("push_id_str", [](const char* id) { ImGui::PushID(id); }, py::arg("id"));
    ig.def("push_id_int", [](int id) { ImGui::PushID(id); }, py::arg("id"));
    ig.def("pop_id",      []() { ImGui::PopID(); });

    // ---------- Style --------------------------------------------------
    ig.def("push_style_color", [](int idx, float r, float g, float b, float a) {
        ImGui::PushStyleColor(idx, ImVec4(r, g, b, a));
    }, py::arg("idx"), py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a"));
    ig.def("pop_style_color", [](int count) { ImGui::PopStyleColor(count); }, py::arg("count") = 1);

    ig.def("push_style_var_float", [](int idx, float val) { ImGui::PushStyleVar(idx, val); },
           py::arg("idx"), py::arg("value"));
    ig.def("push_style_var_vec2", [](int idx, float x, float y) {
        ImGui::PushStyleVar(idx, ImVec2(x, y));
    }, py::arg("idx"), py::arg("x"), py::arg("y"));
    ig.def("pop_style_var", [](int count) { ImGui::PopStyleVar(count); }, py::arg("count") = 1);

    ig.def("style_colors_dark",    []() { ImGui::StyleColorsDark(); });
    ig.def("style_colors_light",   []() { ImGui::StyleColorsLight(); });
    ig.def("style_colors_classic", []() { ImGui::StyleColorsClassic(); });

    // ---------- Cursor / Viewport info ---------------------------------
    ig.def("get_cursor_pos", []() {
        auto p = ImGui::GetCursorPos();
        return py::make_tuple(p.x, p.y);
    });
    ig.def("set_cursor_pos", [](float x, float y) { ImGui::SetCursorPos(ImVec2(x, y)); },
           py::arg("x"), py::arg("y"));
    ig.def("get_content_region_avail", []() {
        auto s = ImGui::GetContentRegionAvail();
        return py::make_tuple(s.x, s.y);
    });
    ig.def("get_window_size", []() {
        auto s = ImGui::GetWindowSize();
        return py::make_tuple(s.x, s.y);
    });
    ig.def("get_window_pos", []() {
        auto p = ImGui::GetWindowPos();
        return py::make_tuple(p.x, p.y);
    });
    ig.def("set_next_window_pos", [](float x, float y, int cond) {
        ImGui::SetNextWindowPos(ImVec2(x, y), cond);
    }, py::arg("x"), py::arg("y"), py::arg("cond") = 0);
    ig.def("set_next_window_size", [](float w, float h, int cond) {
        ImGui::SetNextWindowSize(ImVec2(w, h), cond);
    }, py::arg("width"), py::arg("height"), py::arg("cond") = 0);

    // ---------- Input state queries ------------------------------------
    ig.def("is_item_hovered",    []() { return ImGui::IsItemHovered(); });
    ig.def("is_item_active",     []() { return ImGui::IsItemActive(); });
    ig.def("is_item_clicked",    [](int btn) { return ImGui::IsItemClicked(btn); },
           py::arg("button") = 0);
    ig.def("is_item_focused",    []() { return ImGui::IsItemFocused(); });
    ig.def("is_item_edited",     []() { return ImGui::IsItemEdited(); });
    ig.def("is_item_deactivated_after_edit", []() { return ImGui::IsItemDeactivatedAfterEdit(); });

    ig.def("get_mouse_pos", []() {
        auto p = ImGui::GetMousePos();
        return py::make_tuple(p.x, p.y);
    });
    ig.def("is_mouse_clicked",     [](int btn) { return ImGui::IsMouseClicked(btn); },
           py::arg("button") = 0);
    ig.def("is_mouse_double_clicked", [](int btn) { return ImGui::IsMouseDoubleClicked(btn); },
           py::arg("button") = 0);
    ig.def("is_mouse_dragging",    [](int btn, float thresh) { return ImGui::IsMouseDragging(btn, thresh); },
           py::arg("button") = 0, py::arg("lock_threshold") = -1.f);
    ig.def("get_mouse_drag_delta", [](int btn, float thresh) {
        auto d = ImGui::GetMouseDragDelta(btn, thresh);
        return py::make_tuple(d.x, d.y);
    }, py::arg("button") = 0, py::arg("lock_threshold") = -1.f);

    ig.def("is_key_pressed", [](int key) { return ImGui::IsKeyPressed(static_cast<ImGuiKey>(key)); },
           py::arg("key"));
    ig.def("is_key_down", [](int key) { return ImGui::IsKeyDown(static_cast<ImGuiKey>(key)); },
           py::arg("key"));

    // ---------- Drawing (DrawList) -------------------------------------
    ig.def("draw_line", [](float x1, float y1, float x2, float y2, unsigned int col, float thickness) {
        ImGui::GetWindowDrawList()->AddLine(ImVec2(x1, y1), ImVec2(x2, y2), col, thickness);
    }, py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
       py::arg("col") = IM_COL32(255,255,255,255), py::arg("thickness") = 1.f);

    // Background draw list (renders behind all ImGui windows)
    ig.def("bg_draw_image", [](uintptr_t tex_id, float x1, float y1, float x2, float y2,
                               float uv0x, float uv0y, float uv1x, float uv1y) {
        ImGui::GetBackgroundDrawList()->AddImage((ImTextureID)tex_id,
            ImVec2(x1, y1), ImVec2(x2, y2), ImVec2(uv0x, uv0y), ImVec2(uv1x, uv1y));
    }, py::arg("texture_id"), py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
       py::arg("uv0x") = 0.f, py::arg("uv0y") = 0.f, py::arg("uv1x") = 1.f, py::arg("uv1y") = 1.f);

    ig.def("get_display_size", []() {
        auto& io = ImGui::GetIO();
        return py::make_tuple(io.DisplaySize.x, io.DisplaySize.y);
    });

    ig.def("draw_rect", [](float x1, float y1, float x2, float y2, unsigned int col, float rounding, float thickness) {
        ImGui::GetWindowDrawList()->AddRect(ImVec2(x1, y1), ImVec2(x2, y2), col, rounding, 0, thickness);
    }, py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
       py::arg("col") = IM_COL32(255,255,255,255), py::arg("rounding") = 0.f, py::arg("thickness") = 1.f);

    ig.def("draw_rect_filled", [](float x1, float y1, float x2, float y2, unsigned int col, float rounding) {
        ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(x1, y1), ImVec2(x2, y2), col, rounding);
    }, py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
       py::arg("col") = IM_COL32(255,255,255,255), py::arg("rounding") = 0.f);

    ig.def("draw_circle", [](float cx, float cy, float r, unsigned int col, int segs, float thickness) {
        ImGui::GetWindowDrawList()->AddCircle(ImVec2(cx, cy), r, col, segs, thickness);
    }, py::arg("cx"), py::arg("cy"), py::arg("radius"),
       py::arg("col") = IM_COL32(255,255,255,255), py::arg("segments") = 0, py::arg("thickness") = 1.f);

    ig.def("draw_circle_filled", [](float cx, float cy, float r, unsigned int col, int segs) {
        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(cx, cy), r, col, segs);
    }, py::arg("cx"), py::arg("cy"), py::arg("radius"),
       py::arg("col") = IM_COL32(255,255,255,255), py::arg("segments") = 0);

    ig.def("draw_text", [](float x, float y, unsigned int col, const char* text) {
        ImGui::GetWindowDrawList()->AddText(ImVec2(x, y), col, text);
    }, py::arg("x"), py::arg("y"), py::arg("col"), py::arg("text"));

    // ---------- Utility ------------------------------------------------
    ig.def("get_io_framerate",  []() { return ImGui::GetIO().Framerate; });
    ig.def("get_io_delta_time", []() { return ImGui::GetIO().DeltaTime; });
    ig.def("get_time",          []() { return ImGui::GetTime(); });
    ig.def("get_frame_count",   []() { return ImGui::GetFrameCount(); });

    ig.def("progress_bar", [](float fraction, float w, float h, const char* overlay) {
        ImGui::ProgressBar(fraction, ImVec2(w, h), overlay && overlay[0] ? overlay : nullptr);
    }, py::arg("fraction"), py::arg("width") = -1.f, py::arg("height") = 0.f, py::arg("overlay") = "");

    ig.def("image", [](uintptr_t tex_id, float w, float h, float uv0x, float uv0y, float uv1x, float uv1y) {
        ImGui::Image((ImTextureID)tex_id, ImVec2(w, h), ImVec2(uv0x, uv0y), ImVec2(uv1x, uv1y));
    }, py::arg("texture_id"), py::arg("width"), py::arg("height"),
       py::arg("uv0x") = 0.f, py::arg("uv0y") = 0.f, py::arg("uv1x") = 1.f, py::arg("uv1y") = 1.f);

    ig.def("image_button", [](const char* id, uintptr_t tex_id, float w, float h) {
        return ImGui::ImageButton(id, (ImTextureID)tex_id, ImVec2(w, h));
    }, py::arg("id"), py::arg("texture_id"), py::arg("width"), py::arg("height"));

    // ---------- Plot helpers -------------------------------------------
    ig.def("plot_lines", [](const char* label, std::vector<float> vals, int offset,
                            const char* overlay, float scale_min, float scale_max, float w, float h) {
        ImGui::PlotLines(label, vals.data(), static_cast<int>(vals.size()),
                         offset, overlay && overlay[0] ? overlay : nullptr,
                         scale_min, scale_max, ImVec2(w, h));
    }, py::arg("label"), py::arg("values"), py::arg("offset") = 0, py::arg("overlay") = "",
       py::arg("scale_min") = FLT_MAX, py::arg("scale_max") = FLT_MAX,
       py::arg("width") = 0.f, py::arg("height") = 0.f);

    ig.def("plot_histogram", [](const char* label, std::vector<float> vals, int offset,
                                const char* overlay, float scale_min, float scale_max, float w, float h) {
        ImGui::PlotHistogram(label, vals.data(), static_cast<int>(vals.size()),
                             offset, overlay && overlay[0] ? overlay : nullptr,
                             scale_min, scale_max, ImVec2(w, h));
    }, py::arg("label"), py::arg("values"), py::arg("offset") = 0, py::arg("overlay") = "",
       py::arg("scale_min") = FLT_MAX, py::arg("scale_max") = FLT_MAX,
       py::arg("width") = 0.f, py::arg("height") = 0.f);

    // ---------- Misc ---------------------------------------------------
    ig.def("show_demo_window", []() { ImGui::ShowDemoWindow(); });
    ig.def("show_metrics_window", []() { ImGui::ShowMetricsWindow(); });

    // ---------- Docking ------------------------------------------------
    ig.def("dock_space_over_viewport", [](int flags) -> unsigned int {
        return (unsigned int)ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), flags);
    }, py::arg("flags") = 0,
       "Create a full-viewport dockspace. Returns the dockspace ID.");

    ig.def("dock_space", [](unsigned int id, float w, float h, int flags) {
        return (unsigned int)ImGui::DockSpace(id, ImVec2(w, h), flags);
    }, py::arg("id"), py::arg("width") = 0.f, py::arg("height") = 0.f,
       py::arg("flags") = 0);

    ig.def("set_next_window_dock_id", [](unsigned int id, int cond) {
        ImGui::SetNextWindowDockID(id, cond);
    }, py::arg("dock_id"), py::arg("cond") = 0);

    // ---------- ID helpers ---------------------------------------------
    ig.def("get_id", [](const char* str_id) -> unsigned int {
        return ImGui::GetID(str_id);
    }, py::arg("str_id"));

    // ---------- DockBuilder (programmatic layout) ----------------------
    ig.def("dock_builder_add_node", [](unsigned int id, int flags) -> unsigned int {
        return ImGui::DockBuilderAddNode(id, flags);
    }, py::arg("node_id") = 0, py::arg("flags") = 0);

    ig.def("dock_builder_remove_node", [](unsigned int id) {
        ImGui::DockBuilderRemoveNode(id);
    }, py::arg("node_id"));

    ig.def("dock_builder_set_node_size", [](unsigned int id, float w, float h) {
        ImGui::DockBuilderSetNodeSize(id, ImVec2(w, h));
    }, py::arg("node_id"), py::arg("width"), py::arg("height"));

    ig.def("dock_builder_set_node_pos", [](unsigned int id, float x, float y) {
        ImGui::DockBuilderSetNodePos(id, ImVec2(x, y));
    }, py::arg("node_id"), py::arg("x"), py::arg("y"));

    ig.def("dock_builder_split_node", [](unsigned int node_id, int split_dir,
                                         float size_ratio) -> py::tuple {
        ImGuiID id_at_dir = 0, id_opposite = 0;
        ImGui::DockBuilderSplitNode(node_id, static_cast<ImGuiDir>(split_dir),
                                    size_ratio, &id_at_dir, &id_opposite);
        return py::make_tuple((unsigned int)id_at_dir, (unsigned int)id_opposite);
    }, py::arg("node_id"), py::arg("split_dir"), py::arg("size_ratio"),
       "Split a dock node. Returns (id_at_dir, id_at_opposite). "
       "Directions: 0=Left, 1=Right, 2=Up, 3=Down.");

    ig.def("dock_builder_dock_window", [](const char* window_name, unsigned int node_id) {
        ImGui::DockBuilderDockWindow(window_name, node_id);
    }, py::arg("window_name"), py::arg("node_id"));

    ig.def("dock_builder_finish", [](unsigned int node_id) {
        ImGui::DockBuilderFinish(node_id);
    }, py::arg("node_id"));

    ig.def("dock_builder_get_node", [](unsigned int node_id) -> bool {
        return ImGui::DockBuilderGetNode(node_id) != nullptr;
    }, py::arg("node_id"), "Returns True if the dock node exists.");

    ig.def("col32", [](int r, int g, int b, int a) -> unsigned int {
        return IM_COL32(r, g, b, a);
    }, py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a") = 255,
       "Create a packed RGBA color (0-255 per channel).");
}
