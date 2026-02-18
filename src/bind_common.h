#pragma once

// ── Vultorch binding declarations ─────────────────────────────────────
// Each function registers a group of pybind11 bindings onto the given module.
// Split from the monolithic bindings.cpp for maintainability and to make it
// easy to add new binding groups (e.g. custom renderers, new widget sets).

#include <pybind11/pybind11.h>

namespace py = pybind11;

/// Engine class + CUDA tensor interop + SceneRenderer bindings.
void bind_engine(py::module_& m);

/// ImGui interactive widgets: text, buttons, sliders, inputs, colors,
/// combos, trees, tabs, selectable, progress, plots, images.
void bind_imgui_widgets(py::module_& ig);

/// ImGui windowing, layout, containers: begin/end, child windows, tables,
/// menus, popups, tooltips, style, ID stack, docking, DockBuilder.
void bind_imgui_layout(py::module_& ig);

/// ImGui drawing primitives (DrawList), input-state queries (mouse, keyboard,
/// item hover), and utility functions (framerate, time, demo windows).
void bind_imgui_draw(py::module_& ig);

/// Image I/O (imread / imwrite) via stb_image.
void bind_io(py::module_& m);
