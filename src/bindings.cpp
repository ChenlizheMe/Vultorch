// ── Vultorch module entry point ────────────────────────────────────────
// Delegates to per-category binding files for maintainability:
//   bind_engine.cpp       — Engine class, CUDA tensor interop, SceneRenderer
//   bind_imgui_widgets.cpp — text, buttons, sliders, inputs, colors, plots, …
//   bind_imgui_layout.cpp  — windows, layout, tables, menus, docking, style, …
//   bind_imgui_draw.cpp    — DrawList primitives, mouse/key queries, utility

#include "bind_common.h"

PYBIND11_MODULE(_vultorch, m) {
    m.doc() = "Vultorch – Vulkan + PyTorch zero-copy interop with ImGui";

    // ── Core engine (Vulkan, CUDA tensor interop, SceneRenderer) ───
    bind_engine(m);

    // ── CUDA availability flag ─────────────────────────────────────
    m.attr("HAS_CUDA") =
#ifdef VULTORCH_HAS_CUDA
        true;
#else
        false;
#endif

    // ── ImGui sub-module ───────────────────────────────────────────
    auto ig = m.def_submodule("ui", "Dear ImGui bindings for Python");
    bind_imgui_widgets(ig);
    bind_imgui_layout(ig);
    bind_imgui_draw(ig);
}
