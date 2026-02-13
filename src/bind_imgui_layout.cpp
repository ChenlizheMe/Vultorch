// ── ImGui layout and windowing bindings ────────────────────────────────
// Windows, child windows, layout/spacing, groups, tables, menus, popups,
// tooltips, ID stack, style, cursor/viewport, docking, DockBuilder.

#include "bind_common.h"
#include <pybind11/stl.h>
#include <imgui.h>
#include <imgui_internal.h>   // DockBuilder APIs
#include <string>

void bind_imgui_layout(py::module_& ig) {

    // ── Windows ────────────────────────────────────────────────────
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

    // ── Layout / Spacing ───────────────────────────────────────────
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

    // ── Tables ─────────────────────────────────────────────────────
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

    // ── Menu bar ───────────────────────────────────────────────────
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

    // ── Popups / Modals ────────────────────────────────────────────
    ig.def("open_popup", [](const char* id) { ImGui::OpenPopup(id); }, py::arg("id"));
    ig.def("begin_popup", [](const char* id, int flags) {
        return ImGui::BeginPopup(id, flags);
    }, py::arg("id"), py::arg("flags") = 0);
    ig.def("begin_popup_modal", [](const char* name, int flags) {
        return ImGui::BeginPopupModal(name, nullptr, flags);
    }, py::arg("name"), py::arg("flags") = 0);
    ig.def("end_popup", []() { ImGui::EndPopup(); });
    ig.def("close_current_popup", []() { ImGui::CloseCurrentPopup(); });

    // ── Tooltips ───────────────────────────────────────────────────
    ig.def("begin_tooltip", []() { return ImGui::BeginTooltip(); });
    ig.def("end_tooltip",   []() { ImGui::EndTooltip(); });
    ig.def("set_tooltip", [](const char* t) { ImGui::SetTooltip("%s", t); }, py::arg("text"));

    // ── ID stack ───────────────────────────────────────────────────
    ig.def("push_id_str", [](const char* id) { ImGui::PushID(id); }, py::arg("id"));
    ig.def("push_id_int", [](int id) { ImGui::PushID(id); }, py::arg("id"));
    ig.def("pop_id",      []() { ImGui::PopID(); });

    // ── Style ──────────────────────────────────────────────────────
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

    // ── Cursor / Viewport info ─────────────────────────────────────
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

    // ── Docking ────────────────────────────────────────────────────
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

    ig.def("get_id", [](const char* str_id) -> unsigned int {
        return ImGui::GetID(str_id);
    }, py::arg("str_id"));

    // ── DockBuilder (programmatic layout) ──────────────────────────
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
}
