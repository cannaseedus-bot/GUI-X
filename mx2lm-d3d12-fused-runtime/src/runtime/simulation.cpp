#include "simulation.h"
#include "../buffers/buffer_manager.h"
#include <cstdio>
#include <csignal>

namespace mx2lm {

static volatile bool g_stop = false;
static void HandleSigInt(int) { g_stop = true; }

// ── Init ──────────────────────────────────────────────────────

void Simulation::Init(const Config& cfg) {
    m_cfg = cfg;

    // ── DX12 device ──────────────────────────────────────────
    m_ctx.Init(cfg.enableDebugLayer);

    // ── GPU buffers ───────────────────────────────────────────
    m_bufs.Init(m_ctx.Device(), cfg);

    // ── Stream manager (32 GB / 4 MB pages = up to 8192 pages) ──
    m_stream.Init(m_ctx.Device(),
                  cfg.vramBudgetBytes / STREAM_PAGE_SIZE);

    // ── Load initial entity state ─────────────────────────────
    if (cfg.scxq2Path) {
        m_loader.Load(cfg.scxq2Path, &m_ctx, &m_bufs);
    } else {
        m_loader.GenerateRandom(cfg.entityCount, &m_ctx, &m_bufs);
    }

    // ── Grid + Fused kernels ──────────────────────────────────
    m_grid .Init(&m_ctx, &m_bufs, cfg);
    m_fused.Init(&m_ctx, &m_bufs, cfg);

    // ── Optional training ─────────────────────────────────────
    if (cfg.enableTraining) {
        AdamWConfig ac;
        ac.lr         = cfg.trainLr;
        ac.paramCount = cfg.entityCount;  // 1 weight per entity (MVP)

        LoRAConfig lc;
        lc.rank = cfg.loraRank;

        m_trainer.Init(&m_ctx, ac, lc, "build/shaders");
    }

    m_running = true;

    printf("[mx2lm] Init complete\n");
    printf("        entities=%u  grid=%ux%ux%u  vram_budget=%.1f GB\n",
           cfg.entityCount, cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ,
           cfg.vramBudgetBytes / (1024.0 * 1024.0 * 1024.0));
    if (cfg.enableTraining)
        printf("        training=on  lr=%.2e  lora_rank=%u\n",
               cfg.trainLr, cfg.loraRank);
}

// ── Single step ───────────────────────────────────────────────

void Simulation::RecordFrame(ID3D12GraphicsCommandList* cmdList) {
    auto& b = m_bufs.Buffers();

    // ── 1. Grid build ─────────────────────────────────────────
    m_grid.BuildGrid(cmdList);
    BufferManager::UAVBarrier(cmdList, b.gridIndices.Resource());

    // ── 2. Fused kernel ───────────────────────────────────────
    m_fused.Dispatch(cmdList);
    BufferManager::UAVBarrier(cmdList, b.force    .Resource());
    BufferManager::UAVBarrier(cmdList, b.signalOut.Resource());

    // ── 3. Training step (if enabled) ────────────────────────
    if (m_cfg.enableTraining) {
        // UAV barrier before reading signals as grad proxy
        BufferManager::UAVBarrier(cmdList, b.signalOut.Resource());

        m_trainer.StepWeights(cmdList, (uint32_t)m_step + 1u);
        BufferManager::UAVBarrier(cmdList, m_trainer.WeightsBuf());

        m_trainer.StepLoRA(cmdList, (uint32_t)m_step + 1u);
        BufferManager::UAVBarrier(cmdList, m_trainer.LoRAaBuf());
    }
}

void Simulation::Step() {
    m_ctx.ResetCommandList();
    RecordFrame(m_ctx.CmdList());
    m_ctx.ExecuteAndWait();

    ++m_step;

    if (m_step % 100 == 0) {
        printf("[mx2lm] step=%llu  stream_pages=%u\n",
               (unsigned long long)m_step,
               m_stream.ResidentCount());
    }
}

// ── Run loop ──────────────────────────────────────────────────

void Simulation::Run() {
    std::signal(SIGINT, HandleSigInt);
    printf("[mx2lm] Running... (Ctrl+C to stop)\n\n");

    while (!g_stop) {
        Step();
        if (m_cfg.maxSteps > 0 && m_step >= m_cfg.maxSteps)
            break;
    }

    printf("\n[mx2lm] Stopped at step %llu\n", (unsigned long long)m_step);
}

// ── Shutdown ──────────────────────────────────────────────────

void Simulation::Shutdown() {
    m_ctx.WaitIdle();
    m_running = false;
    printf("[mx2lm] Shutdown complete.\n");
}

} // namespace mx2lm
