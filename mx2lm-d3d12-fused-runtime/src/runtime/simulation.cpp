#include "simulation.h"
#include <cstdio>
#include <csignal>

namespace mx2lm {

static volatile bool g_stop = false;
static void HandleSigInt(int) { g_stop = true; }

// ── Init ──────────────────────────────────────────────────────

void Simulation::Init(const Config& cfg) {
    m_cfg = cfg;

    m_ctx.Init(cfg.enableDebugLayer);
    m_bufs.Init(m_ctx.Device(), cfg);

    // Load initial state
    if (cfg.scxq2Path) {
        m_loader.Load(cfg.scxq2Path, &m_ctx, &m_bufs);
    } else {
        m_loader.GenerateRandom(cfg.entityCount, &m_ctx, &m_bufs);
    }

    m_grid .Init(&m_ctx, &m_bufs, cfg);
    m_fused.Init(&m_ctx, &m_bufs, cfg);

    m_running = true;
    printf("[mx2lm] Simulation initialized: %u entities, %u grid cells\n",
           cfg.entityCount, cfg.gridCellCount());
}

// ── Single Step ───────────────────────────────────────────────

void Simulation::RecordFrame(ID3D12GraphicsCommandList* cmdList) {
    // 1. Build spatial grid
    m_grid.BuildGrid(cmdList);

    // 2. UAV barrier before fused kernel reads grid
    BufferManager::UAVBarrier(cmdList, m_bufs.Buffers().gridIndices.Resource());

    // 3. Fused kernel
    m_fused.Dispatch(cmdList);

    // 4. UAV barrier on outputs
    BufferManager::UAVBarrier(cmdList, m_bufs.Buffers().force.Resource());
    BufferManager::UAVBarrier(cmdList, m_bufs.Buffers().signalOut.Resource());
}

void Simulation::Step() {
    m_ctx.ResetCommandList();
    RecordFrame(m_ctx.CmdList());
    m_ctx.ExecuteAndWait();

    ++m_step;

    if (m_step % 100 == 0)
        printf("[mx2lm] Step %llu\n", (unsigned long long)m_step);
}

// ── Run Loop ──────────────────────────────────────────────────

void Simulation::Run() {
    std::signal(SIGINT, HandleSigInt);
    printf("[mx2lm] Running... (Ctrl+C to stop)\n");

    while (!g_stop) {
        Step();
        if (m_cfg.maxSteps > 0 && m_step >= m_cfg.maxSteps)
            break;
    }

    printf("[mx2lm] Stopped at step %llu\n", (unsigned long long)m_step);
}

// ── Shutdown ──────────────────────────────────────────────────

void Simulation::Shutdown() {
    m_ctx.WaitIdle();
    m_running = false;
}

} // namespace mx2lm
