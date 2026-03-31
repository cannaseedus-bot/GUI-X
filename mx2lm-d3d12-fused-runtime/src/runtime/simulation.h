#pragma once
#include "../dx12_context.h"
#include "../buffers/buffer_manager.h"
#include "../kernels/fused_dispatch.h"
#include "../kernels/grid_dispatch.h"
#include "../scxq2/scxq2_loader.h"
#include "config.h"
#include <cstdint>

namespace mx2lm {

class Simulation {
public:
    void Init(const Config& cfg);
    void Step();
    void Run();    // runs until maxSteps or Ctrl+C
    void Shutdown();

    uint64_t StepCount() const { return m_step; }

private:
    Config         m_cfg;
    DX12Context    m_ctx;
    BufferManager  m_bufs;
    SCXQ2Loader    m_loader;
    GridDispatch   m_grid;
    FusedDispatch  m_fused;

    uint64_t       m_step = 0;
    bool           m_running = false;

    void RecordFrame(ID3D12GraphicsCommandList* cmdList);
};

} // namespace mx2lm
