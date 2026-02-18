# ===========================================================================
# external/CMakeLists.txt
# Wraps all third-party dependencies and exports clean targets to the parent.
# ===========================================================================

# ---- pybind11 -------------------------------------------------------------
# Force pybind11 to use the same Python that the top-level project found,
# preventing it from falling back to FindPythonInterp and picking up a
# different interpreter (e.g. base conda instead of the active env).
set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "" FORCE)
set(PYBIND11_PYTHON_VERSION "${Python3_VERSION}" CACHE STRING "" FORCE)
add_subdirectory(pybind11 EXCLUDE_FROM_ALL)

# ---- SDL3 (static) --------------------------------------------------------
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON  CACHE BOOL "" FORCE)
set(SDL_TEST   OFF CACHE BOOL "" FORCE)
add_subdirectory(SDL EXCLUDE_FROM_ALL)

# ---- Dear ImGui (no CMakeLists.txt – we build it ourselves) ----------------
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui")

add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)

target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

target_compile_definitions(imgui PUBLIC SDL_MAIN_HANDLED)
# SDL3 3.2.0 compatibility: disable legacy macro poisoning so we can redefine them
target_compile_definitions(imgui PUBLIC SDL_DISABLE_OLD_NAMES)
# Now manually map old names to new standard C++ bool
target_compile_definitions(imgui PUBLIC
    SDL_FALSE=false
    SDL_TRUE=true
)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan SDL3-static)

# ---- stb (header-only image I/O) ------------------------------------------
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/stb")
