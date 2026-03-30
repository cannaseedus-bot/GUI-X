#pragma once
#include <cstdint>

namespace mx2lm {

struct Config {
    // ── Entity / Grid ─────────────────────────────────────────
    uint32_t entityCount    = 4096;
    uint32_t gridDimX       = 64;
    uint32_t gridDimY       = 64;
    uint32_t gridDimZ       = 64;
    float    cellSize       = 1.0f;
    float    gridOriginX    = -32.0f;
    float    gridOriginY    = -32.0f;
    float    gridOriginZ    = -32.0f;

    // ── Kernel Parameters ─────────────────────────────────────
    uint32_t neighborCap    = 64;
    uint32_t numExperts     = 8;
    float    forceMax       = 1e4f;
    float    eventThreshold = 100.0f;

    // ── Runtime ───────────────────────────────────────────────
    uint32_t maxSteps       = 0;        // 0 = unlimited
    bool     enableReadback = false;    // debug only
    bool     enableViz      = false;    // debug overlay
    bool     enableDebugLayer = false;  // D3D12 debug layer

    // ── SCXQ2 ─────────────────────────────────────────────────
    const char* scxq2Path   = nullptr;  // nullptr = generate random initial state

    // ── Computed ──────────────────────────────────────────────
    uint32_t gridCellCount() const {
        return gridDimX * gridDimY * gridDimZ;
    }
};

} // namespace mx2lm
