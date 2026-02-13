// ── ImGui widget bindings ──────────────────────────────────────────────
// Text, buttons, sliders, drag inputs, input fields, color pickers,
// combo/listbox, tree/collapsing, selectable, tabs, progress, plots, images.

#include "bind_common.h"
#include <pybind11/stl.h>
#include <imgui.h>
#include <cfloat>
#include <string>
#include <vector>

void bind_imgui_widgets(py::module_& ig) {

    // ── Text / Labels ──────────────────────────────────────────────
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

    // ── Buttons ────────────────────────────────────────────────────
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

    // ── Checkbox ───────────────────────────────────────────────────
    ig.def("checkbox", [](const char* label, bool v) {
        ImGui::Checkbox(label, &v);
        return v;
    }, py::arg("label"), py::arg("value"));

    // ── Sliders ────────────────────────────────────────────────────
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

    // ── Drag inputs ────────────────────────────────────────────────
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

    // ── Input fields ───────────────────────────────────────────────
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

    // ── Color ──────────────────────────────────────────────────────
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

    // ── Combo / Listbox ────────────────────────────────────────────
    ig.def("combo", [](const char* label, int current, std::vector<std::string> items) {
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

    // ── Tree / Collapsing headers ──────────────────────────────────
    ig.def("tree_node", [](const char* label) { return ImGui::TreeNode(label); }, py::arg("label"));
    ig.def("tree_pop", []() { ImGui::TreePop(); });
    ig.def("collapsing_header", [](const char* label, int flags) {
        return ImGui::CollapsingHeader(label, flags);
    }, py::arg("label"), py::arg("flags") = 0);

    // ── Selectable ─────────────────────────────────────────────────
    ig.def("selectable", [](const char* label, bool selected, int flags, float w, float h) {
        ImGui::Selectable(label, &selected, flags, ImVec2(w, h));
        return selected;
    }, py::arg("label"), py::arg("selected") = false, py::arg("flags") = 0,
       py::arg("width") = 0.f, py::arg("height") = 0.f);

    // ── Tabs ───────────────────────────────────────────────────────
    ig.def("begin_tab_bar", [](const char* id, int flags) {
        return ImGui::BeginTabBar(id, flags);
    }, py::arg("id"), py::arg("flags") = 0);
    ig.def("end_tab_bar", []() { ImGui::EndTabBar(); });

    ig.def("begin_tab_item", [](const char* label) {
        return ImGui::BeginTabItem(label);
    }, py::arg("label"));
    ig.def("end_tab_item", []() { ImGui::EndTabItem(); });

    // ── Progress bar ───────────────────────────────────────────────
    ig.def("progress_bar", [](float fraction, float w, float h, const char* overlay) {
        ImGui::ProgressBar(fraction, ImVec2(w, h), overlay && overlay[0] ? overlay : nullptr);
    }, py::arg("fraction"), py::arg("width") = -1.f, py::arg("height") = 0.f, py::arg("overlay") = "");

    // ── Image / Image button ───────────────────────────────────────
    ig.def("image", [](uintptr_t tex_id, float w, float h, float uv0x, float uv0y, float uv1x, float uv1y) {
        ImGui::Image((ImTextureID)tex_id, ImVec2(w, h), ImVec2(uv0x, uv0y), ImVec2(uv1x, uv1y));
    }, py::arg("texture_id"), py::arg("width"), py::arg("height"),
       py::arg("uv0x") = 0.f, py::arg("uv0y") = 0.f, py::arg("uv1x") = 1.f, py::arg("uv1y") = 1.f);

    ig.def("image_button", [](const char* id, uintptr_t tex_id, float w, float h) {
        return ImGui::ImageButton(id, (ImTextureID)tex_id, ImVec2(w, h));
    }, py::arg("id"), py::arg("texture_id"), py::arg("width"), py::arg("height"));

    // ── Plot helpers ───────────────────────────────────────────────
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
}
