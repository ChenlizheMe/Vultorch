// ── ImGui drawing primitives, input queries, and utility bindings ──────
// DrawList operations, background draw list, mouse/keyboard state,
// framerate/time queries, demo windows, color helpers.

#include "bind_common.h"
#include <imgui.h>

void bind_imgui_draw(py::module_& ig) {

    // ── Drawing (WindowDrawList) ───────────────────────────────────
    ig.def("draw_line", [](float x1, float y1, float x2, float y2, unsigned int col, float thickness) {
        ImGui::GetWindowDrawList()->AddLine(ImVec2(x1, y1), ImVec2(x2, y2), col, thickness);
    }, py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
       py::arg("col") = IM_COL32(255,255,255,255), py::arg("thickness") = 1.f);

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

    // ── Background draw list ───────────────────────────────────────
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

    // ── Item state queries ─────────────────────────────────────────
    ig.def("is_item_hovered",    []() { return ImGui::IsItemHovered(); });
    ig.def("is_item_active",     []() { return ImGui::IsItemActive(); });
    ig.def("is_item_clicked",    [](int btn) { return ImGui::IsItemClicked(btn); },
           py::arg("button") = 0);
    ig.def("is_item_focused",    []() { return ImGui::IsItemFocused(); });
    ig.def("is_item_edited",     []() { return ImGui::IsItemEdited(); });
    ig.def("is_item_deactivated_after_edit", []() { return ImGui::IsItemDeactivatedAfterEdit(); });

    // ── Mouse state ────────────────────────────────────────────────
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

    // ── Keyboard state ─────────────────────────────────────────────
    ig.def("is_key_pressed", [](int key) { return ImGui::IsKeyPressed(static_cast<ImGuiKey>(key)); },
           py::arg("key"));
    ig.def("is_key_down", [](int key) { return ImGui::IsKeyDown(static_cast<ImGuiKey>(key)); },
           py::arg("key"));

    // ── Utility ────────────────────────────────────────────────────
    ig.def("get_io_framerate",  []() { return ImGui::GetIO().Framerate; });
    ig.def("get_io_delta_time", []() { return ImGui::GetIO().DeltaTime; });
    ig.def("get_time",          []() { return ImGui::GetTime(); });
    ig.def("get_frame_count",   []() { return ImGui::GetFrameCount(); });

    // ── Misc ───────────────────────────────────────────────────────
    ig.def("show_demo_window", []() { ImGui::ShowDemoWindow(); });
    ig.def("show_metrics_window", []() { ImGui::ShowMetricsWindow(); });

    ig.def("col32", [](int r, int g, int b, int a) -> unsigned int {
        return IM_COL32(r, g, b, a);
    }, py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a") = 255,
       "Create a packed RGBA color (0-255 per channel).");
}
