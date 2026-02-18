cmake_minimum_required(VERSION 3.25)
project(vultorch LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Options
# ============================================================================

option(VULTORCH_BUILD_DOCS "Build tutorial docs with mkdocs (requires mkdocs)" ON)

# ============================================================================
# Dependencies: System
# ============================================================================

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Vulkan REQUIRED)

# ── CUDA (optional: enables GPU-GPU zero-copy) ─────────────────────────
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    set(VULTORCH_HAS_CUDA ON)
    message(STATUS "[vultorch] CUDA ${CUDAToolkit_VERSION} — GPU zero-copy enabled")
else()
    set(VULTORCH_HAS_CUDA OFF)
    message(STATUS "[vultorch] CUDA not found — building CPU-only (host staging)")
endif()

# ============================================================================
# Dependencies: external/ (pybind11, SDL3, Dear ImGui)
# ============================================================================

add_subdirectory(external)

# ============================================================================
# Target: _vultorch (pybind11 extension module)
# ============================================================================

pybind11_add_module(_vultorch
    src/engine.cpp
    src/bindings.cpp
    src/bind_engine.cpp
    src/bind_imgui_widgets.cpp
    src/bind_imgui_layout.cpp
    src/bind_imgui_draw.cpp
    src/bind_io.cpp
    src/tensor_texture.cpp
    src/scene_renderer.cpp
)
target_include_directories(_vultorch PRIVATE src)
target_link_libraries(_vultorch PRIVATE imgui Vulkan::Vulkan SDL3-static stb)

# ── CUDA support ────────────────────────────────────────────────────────
if(VULTORCH_HAS_CUDA)
    target_link_libraries(_vultorch PRIVATE CUDA::cuda_driver)
    target_compile_definitions(_vultorch PRIVATE VULTORCH_HAS_CUDA=1)
    if(WIN32)
        target_link_libraries(_vultorch PRIVATE Cfgmgr32)
    endif()
endif()

# ============================================================================
# Shaders: GLSL → SPIR-V → C header
# ============================================================================

find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
if(NOT GLSLC)
    find_program(GLSLC glslangValidator HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
endif()

if(GLSLC)
    message(STATUS "[vultorch] Shader compiler: ${GLSLC}")
    set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders")
    set(SPV_DIR    "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY ${SPV_DIR})

    # plane.vert → plane_vert.spv → plane_vert_spv.h
    add_custom_command(
        OUTPUT  "${SPV_DIR}/plane_vert.spv"
        COMMAND ${GLSLC} "${SHADER_DIR}/plane.vert" -o "${SPV_DIR}/plane_vert.spv"
        DEPENDS "${SHADER_DIR}/plane.vert"
        COMMENT "[vultorch] Compiling plane.vert → SPIR-V"
    )
    add_custom_command(
        OUTPUT  "${SPV_DIR}/plane_vert_spv.h"
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/spv_to_header.py"
                "${SPV_DIR}/plane_vert.spv"
                "${SPV_DIR}/plane_vert_spv.h"
                "plane_vert_spv"
        DEPENDS "${SPV_DIR}/plane_vert.spv"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/spv_to_header.py"
        COMMENT "[vultorch] Embedding plane_vert.spv → C header"
    )

    # plane.frag → plane_frag.spv → plane_frag_spv.h
    add_custom_command(
        OUTPUT  "${SPV_DIR}/plane_frag.spv"
        COMMAND ${GLSLC} "${SHADER_DIR}/plane.frag" -o "${SPV_DIR}/plane_frag.spv"
        DEPENDS "${SHADER_DIR}/plane.frag"
        COMMENT "[vultorch] Compiling plane.frag → SPIR-V"
    )
    add_custom_command(
        OUTPUT  "${SPV_DIR}/plane_frag_spv.h"
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/spv_to_header.py"
                "${SPV_DIR}/plane_frag.spv"
                "${SPV_DIR}/plane_frag_spv.h"
                "plane_frag_spv"
        DEPENDS "${SPV_DIR}/plane_frag.spv"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/spv_to_header.py"
        COMMENT "[vultorch] Embedding plane_frag.spv → C header"
    )

    add_custom_target(vultorch_shaders DEPENDS
        "${SPV_DIR}/plane_vert_spv.h"
        "${SPV_DIR}/plane_frag_spv.h"
    )
    add_dependencies(_vultorch vultorch_shaders)
    target_include_directories(_vultorch PRIVATE "${SPV_DIR}")
else()
    set(FALLBACK_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/generated")
    if(EXISTS "${FALLBACK_SHADER_DIR}/plane_vert_spv.h")
        message(STATUS "[vultorch] Using pre-built shader headers (src/shaders/generated/)")
        target_include_directories(_vultorch PRIVATE "${FALLBACK_SHADER_DIR}")
    else()
        message(WARNING "[vultorch] No shader compiler and no pre-built headers — "
                        "3D scene renderer will not work")
    endif()
endif()

# ============================================================================
# Install (scikit-build-core uses this for wheel layout)
# ============================================================================

install(TARGETS _vultorch
    LIBRARY DESTINATION vultorch
    RUNTIME DESTINATION vultorch
)

# ============================================================================
# Post-build: copy extension module → vultorch/ package directory
# ============================================================================

set(VULTORCH_PKG_DIR "${CMAKE_SOURCE_DIR}/vultorch")

add_custom_command(TARGET _vultorch POST_BUILD
    COMMENT "[vultorch] Cleaning old extensions & copying new module → vultorch/"
    COMMAND "${Python3_EXECUTABLE}" -c
            "import pathlib; [f.unlink() for f in pathlib.Path(r'${VULTORCH_PKG_DIR}').glob('_vultorch*')]"
    COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:_vultorch>" "${VULTORCH_PKG_DIR}/"
)

# ============================================================================
# Target: package_wheel — produce pip-installable .whl in dist/
# ============================================================================

add_custom_target(package_wheel ALL
    COMMENT "[vultorch] Packaging wheel → dist/"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/make_wheel.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    DEPENDS _vultorch
)

# ============================================================================
# Target: docs — build tutorial HTML with mkdocs (optional)
# ============================================================================

if(VULTORCH_BUILD_DOCS)
    find_program(MKDOCS mkdocs)
    if(MKDOCS)
        add_custom_target(docs ALL
            COMMENT "[vultorch] Building tutorial docs (mkdocs) → docs/tutorial/"
            COMMAND "${MKDOCS}" build --clean
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        )
        # docs don't block the build, but run after packaging
        add_dependencies(docs package_wheel)
    else()
        message(STATUS "[vultorch] mkdocs not found — skipping docs "
                       "(install with: pip install mkdocs-material mkdocs-static-i18n)")
    endif()
endif()