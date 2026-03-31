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
    bool     enableReadback = false;
    bool     enableViz      = false;
    bool     enableDebugLayer = false;

    // ── SCXQ2 / State ─────────────────────────────────────────
    const char* scxq2Path   = nullptr;

    // ── VRAM Streaming ────────────────────────────────────────
    uint64_t vramBudgetBytes = 8ull * 1024 * 1024 * 1024;  // 8 GB default

    // ── Training (AdamW + LoRA) ───────────────────────────────
    bool     enableTraining = false;
    float    trainLr        = 1e-4f;
    uint32_t loraRank       = 4;

    // ── Computed ──────────────────────────────────────────────
    uint32_t gridCellCount() const {
        return gridDimX * gridDimY * gridDimZ;
    }
};

} // namespace mx2lm
