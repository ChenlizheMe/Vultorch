"""Tests for vultorch.ui submodule — verify all bound functions exist.

No GPU required — just checks that the C++ bindings expose expected names.

Coverage targets:
  - All widget functions from bind_imgui_widgets.cpp
  - All layout functions from bind_imgui_layout.cpp
  - All draw functions from bind_imgui_draw.cpp
"""

import pytest
from conftest import requires_vultorch


@requires_vultorch
class TestUiWidgetFunctions:
    """Every function exposed in bind_imgui_widgets.cpp should be importable."""

    # Text
    @pytest.mark.parametrize("fn", [
        "text", "text_colored", "text_disabled", "text_wrapped",
        "label_text", "bullet_text",
    ])
    def test_text_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Buttons
    @pytest.mark.parametrize("fn", [
        "button", "small_button", "invisible_button",
        "arrow_button", "radio_button",
    ])
    def test_button_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Inputs — scalar
    @pytest.mark.parametrize("fn", [
        "checkbox",
        "slider_float", "slider_float2", "slider_float3", "slider_float4",
        "slider_int", "slider_angle",
        "drag_float", "drag_float2", "drag_float3", "drag_int",
        "input_float", "input_float2", "input_float3", "input_float4",
        "input_int", "input_text", "input_text_multiline",
    ])
    def test_input_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Colors
    @pytest.mark.parametrize("fn", [
        "color_edit3", "color_edit4", "color_picker3", "color_picker4",
    ])
    def test_color_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Selection
    @pytest.mark.parametrize("fn", [
        "combo", "listbox", "tree_node", "tree_pop",
        "collapsing_header", "selectable",
    ])
    def test_selection_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Tabs
    @pytest.mark.parametrize("fn", [
        "begin_tab_bar", "end_tab_bar", "begin_tab_item", "end_tab_item",
    ])
    def test_tab_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Display
    @pytest.mark.parametrize("fn", [
        "progress_bar", "image", "image_button",
        "plot_lines", "plot_histogram",
    ])
    def test_display_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"


@requires_vultorch
class TestUiLayoutFunctions:
    """Every function exposed in bind_imgui_layout.cpp should be importable."""

    # Windows
    @pytest.mark.parametrize("fn", [
        "begin", "end", "begin_child", "end_child",
    ])
    def test_window_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Layout
    @pytest.mark.parametrize("fn", [
        "separator", "same_line", "new_line", "spacing",
        "dummy", "indent", "unindent",
        "begin_group", "end_group",
        "push_item_width", "pop_item_width",
        "columns", "next_column",
    ])
    def test_layout_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Tables
    @pytest.mark.parametrize("fn", [
        "begin_table", "end_table",
        "table_next_row", "table_next_column", "table_set_column_index",
        "table_setup_column", "table_headers_row",
    ])
    def test_table_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Menus
    @pytest.mark.parametrize("fn", [
        "begin_main_menu_bar", "end_main_menu_bar",
        "begin_menu_bar", "end_menu_bar",
        "begin_menu", "end_menu", "menu_item",
    ])
    def test_menu_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Popups
    @pytest.mark.parametrize("fn", [
        "open_popup", "begin_popup", "begin_popup_modal",
        "end_popup", "close_current_popup",
    ])
    def test_popup_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Tooltips
    @pytest.mark.parametrize("fn", [
        "begin_tooltip", "end_tooltip", "set_tooltip",
    ])
    def test_tooltip_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # ID management
    @pytest.mark.parametrize("fn", [
        "push_id_str", "push_id_int", "pop_id",
    ])
    def test_id_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Style
    @pytest.mark.parametrize("fn", [
        "push_style_color", "pop_style_color",
        "push_style_var_float", "push_style_var_vec2", "pop_style_var",
        "style_colors_dark", "style_colors_light", "style_colors_classic",
    ])
    def test_style_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Cursor / window info
    @pytest.mark.parametrize("fn", [
        "get_cursor_pos", "set_cursor_pos",
        "get_content_region_avail", "get_window_size", "get_window_pos",
        "set_next_window_pos", "set_next_window_size",
    ])
    def test_cursor_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Docking
    @pytest.mark.parametrize("fn", [
        "dock_space_over_viewport", "dock_space",
        "set_next_window_dock_id", "get_id",
        "dock_builder_add_node", "dock_builder_remove_node",
        "dock_builder_set_node_size", "dock_builder_set_node_pos",
        "dock_builder_split_node", "dock_builder_dock_window",
        "dock_builder_finish", "dock_builder_get_node",
    ])
    def test_docking_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"


@requires_vultorch
class TestUiDrawFunctions:
    """Every function exposed in bind_imgui_draw.cpp should be importable."""

    # Drawing
    @pytest.mark.parametrize("fn", [
        "draw_line", "draw_rect", "draw_rect_filled",
        "draw_circle", "draw_circle_filled", "draw_text",
    ])
    def test_draw_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Background
    @pytest.mark.parametrize("fn", [
        "bg_draw_image", "get_display_size",
    ])
    def test_bg_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Item state
    @pytest.mark.parametrize("fn", [
        "is_item_hovered", "is_item_active", "is_item_clicked",
        "is_item_focused", "is_item_edited", "is_item_deactivated_after_edit",
    ])
    def test_item_state_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Mouse
    @pytest.mark.parametrize("fn", [
        "get_mouse_pos", "is_mouse_clicked", "is_mouse_double_clicked",
        "is_mouse_dragging", "get_mouse_drag_delta",
    ])
    def test_mouse_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Keyboard
    @pytest.mark.parametrize("fn", [
        "is_key_pressed", "is_key_down",
    ])
    def test_keyboard_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"

    # Utility
    @pytest.mark.parametrize("fn", [
        "get_io_framerate", "get_io_delta_time",
        "get_time", "get_frame_count",
        "show_demo_window", "show_metrics_window",
        "col32",
    ])
    def test_utility_functions(self, fn):
        from vultorch import ui
        assert callable(getattr(ui, fn)), f"ui.{fn} not callable"
