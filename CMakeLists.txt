cmake_minimum_required(VERSION 3.25)
project(vultorch LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Dependencies: System
# ============================================================================

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Vulkan REQUIRED)

# --- CUDA (for tensor interop via CUDA driver API) ---
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    set(VULTORCH_HAS_CUDA ON)
    message(STATUS "Vultorch: CUDA ${CUDAToolkit_VERSION} found at ${CUDAToolkit_TARGET_DIR}")
else()
    set(VULTORCH_HAS_CUDA OFF)
    message(STATUS "Vultorch: CUDA not found, tensor display disabled")
endif()

# ============================================================================
# Dependencies: external/ (pybind11, SDL3, imgui)
# ============================================================================

add_subdirectory(external)

# ============================================================================
# _vultorch pybind11 module
# ============================================================================

pybind11_add_module(_vultorch
    src/engine.cpp
    src/bindings.cpp
    src/bind_engine.cpp
    src/bind_imgui_widgets.cpp
    src/bind_imgui_layout.cpp
    src/bind_imgui_draw.cpp
)
target_include_directories(_vultorch PRIVATE src)
target_link_libraries(_vultorch PRIVATE imgui Vulkan::Vulkan SDL3-static)

if(VULTORCH_HAS_CUDA)
    target_sources(_vultorch PRIVATE
        src/tensor_texture.cpp
        src/scene_renderer.cpp
    )
    target_link_libraries(_vultorch PRIVATE CUDA::cuda_driver)
    target_compile_definitions(_vultorch PRIVATE VULTORCH_HAS_CUDA=1)

    # ── Shader compilation (GLSL → SPIR-V → C header) ──────────────────
    find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
    if(NOT GLSLC)
        find_program(GLSLC glslangValidator HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
    endif()

    if(GLSLC)
        set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders")
        set(SPV_DIR    "${CMAKE_CURRENT_BINARY_DIR}/shaders")
        file(MAKE_DIRECTORY ${SPV_DIR})

        # Compile plane.vert → plane_vert.spv → plane_vert_spv.h
        add_custom_command(
            OUTPUT  "${SPV_DIR}/plane_vert.spv"
            COMMAND ${GLSLC} "${SHADER_DIR}/plane.vert" -o "${SPV_DIR}/plane_vert.spv"
            DEPENDS "${SHADER_DIR}/plane.vert"
            COMMENT "Compiling plane.vert → SPIR-V"
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
            COMMENT "Embedding plane_vert.spv → C header"
        )

        # Compile plane.frag → plane_frag.spv → plane_frag_spv.h
        add_custom_command(
            OUTPUT  "${SPV_DIR}/plane_frag.spv"
            COMMAND ${GLSLC} "${SHADER_DIR}/plane.frag" -o "${SPV_DIR}/plane_frag.spv"
            DEPENDS "${SHADER_DIR}/plane.frag"
            COMMENT "Compiling plane.frag → SPIR-V"
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
            COMMENT "Embedding plane_frag.spv → C header"
        )

        # Custom target that depends on the generated headers
        add_custom_target(vultorch_shaders DEPENDS
            "${SPV_DIR}/plane_vert_spv.h"
            "${SPV_DIR}/plane_frag_spv.h"
        )
        add_dependencies(_vultorch vultorch_shaders)
        target_include_directories(_vultorch PRIVATE "${SPV_DIR}")
    else()
        message(WARNING "glslc/glslangValidator not found – "
                        "3D scene renderer will not build without shaders")
    endif()

    # Windows: link against OS libraries needed for external memory handles
    if(WIN32)
        target_link_libraries(_vultorch PRIVATE Cfgmgr32)
    endif()
endif()

# ============================================================================
# Install: scikit-build-core picks this up to place .pyd into the wheel
# ============================================================================

install(TARGETS _vultorch
    LIBRARY DESTINATION vultorch
    RUNTIME DESTINATION vultorch
)

# ============================================================================
# Post-build: copy module → vultorch/ and package wheel → dist/
# Always re-packages the wheel on every build invocation.
# ============================================================================

set(VULTORCH_PKG_DIR "${CMAKE_SOURCE_DIR}/vultorch")

# Copy .pyd only when it actually changes
add_custom_command(TARGET _vultorch POST_BUILD
    COMMENT "=== Copying .pyd → vultorch/ ==="
    COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:_vultorch>" "${VULTORCH_PKG_DIR}/"
)

# Always run make_wheel.py (ALL = every build, even if nothing recompiled)
add_custom_target(package_wheel ALL
    COMMENT "=== Packaging wheel → dist/ ==="
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/make_wheel.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    DEPENDS _vultorch
)