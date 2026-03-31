cmake_minimum_required(VERSION 3.20)
project(mx2lm_d3d12_fused_runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Options ───────────────────────────────────────────────────
option(MX2LM_ENABLE_PIX   "Enable PIX GPU capture support" OFF)
option(MX2LM_ENABLE_DEBUG "Enable extra debug output"      OFF)

# ── Platform check ────────────────────────────────────────────
if(NOT WIN32)
    message(FATAL_ERROR "mx2lm-d3d12-fused-runtime requires Windows (D3D12).")
endif()

# ── Source files ──────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/dx12_context.cpp
    src/buffers/buffer_manager.cpp
    src/kernels/fused_dispatch.cpp
    src/kernels/grid_dispatch.cpp
    src/pipeline/pipeline_state.cpp
    src/pipeline/root_signature.cpp
    src/runtime/simulation.cpp
    src/runtime/stream_manager.cpp
    src/scxq2/scxq2_loader.cpp
    src/training/adamw_gpu.cpp
    src/debug/gpu_profiler.cpp
    src/debug/validation.cpp
)

add_executable(mx2lm ${SOURCES})

# ── Include dirs ──────────────────────────────────────────────
target_include_directories(mx2lm PRIVATE
    src
    src/buffers
    src/kernels
    src/pipeline
    src/runtime
    src/scxq2
    src/training
    src/debug
    src/math
)

# ── Windows / D3D12 libraries ─────────────────────────────────
target_link_libraries(mx2lm PRIVATE
    d3d12.lib
    dxgi.lib
    dxguid.lib
    d3dcompiler.lib
    dxcompiler.lib   # DXC — required for inline shader compilation
)

# ── DXC headers (Windows SDK or standalone DXC package) ───────
# Locate dxcapi.h — try Windows SDK first, then env var DXC_DIR
find_path(DXC_INCLUDE_DIR dxcapi.h
    PATHS
        "$ENV{PROGRAMFILES\(X86\)}/Windows Kits/10/Include/10.0.22621.0/um"
        "$ENV{PROGRAMFILES\(X86\)}/Windows Kits/10/Include/10.0.19041.0/um"
        "$ENV{DXC_DIR}/inc"
    DOC "Directory containing dxcapi.h"
)
if(DXC_INCLUDE_DIR)
    target_include_directories(mx2lm PRIVATE ${DXC_INCLUDE_DIR})
    message(STATUS "DXC headers: ${DXC_INCLUDE_DIR}")
else()
    message(WARNING "dxcapi.h not found — set DXC_DIR or install Windows SDK 10.0.19041+")
endif()

# ── DirectXMath (header-only, ships with Windows SDK) ─────────
# No extra find needed — included via <DirectXMath.h>

# ── Compiler flags ────────────────────────────────────────────
target_compile_definitions(mx2lm PRIVATE
    UNICODE
    _UNICODE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
)

if(MX2LM_ENABLE_DEBUG OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(mx2lm PRIVATE MX2LM_DEBUG=1)
endif()

if(MX2LM_ENABLE_PIX)
    target_compile_definitions(mx2lm PRIVATE PIX_ENABLED=1)
    # Add PIX headers/libs here when WinPixEventRuntime is available
endif()

# MSVC-specific
if(MSVC)
    target_compile_options(mx2lm PRIVATE /W4 /WX- /MP /arch:AVX2)
    # Disable warnings for external headers
    target_compile_options(mx2lm PRIVATE /external:anglebrackets /external:W0)
    target_compile_definitions(mx2lm PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# ── Shader compilation output dir ─────────────────────────────
set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

# Post-build: copy compiled shaders to build dir
# (Shaders are pre-compiled by tools/compile_shaders.ps1)
add_custom_command(TARGET mx2lm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo
        "Build complete. Run tools/compile_shaders.ps1 to compile shaders."
)

# ── Install ───────────────────────────────────────────────────
install(TARGETS mx2lm DESTINATION bin)
install(DIRECTORY ${SHADER_OUTPUT_DIR}/ DESTINATION bin/shaders)
