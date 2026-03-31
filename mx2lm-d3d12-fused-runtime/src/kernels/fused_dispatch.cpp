#include "fused_dispatch.h"

namespace mx2lm {

// Root constant layout (must match shader cbuffer RootConstants : register(b0))
struct FusedRootConstants {
    uint32_t entityCount;
    uint32_t gridDimX;
    uint32_t gridDimY;
    uint32_t gridDimZ;
    float    cellSize;
    float    gridOriginX;
    float    gridOriginY;
    float    gridOriginZ;
    float    forceMax;
    uint32_t neighborCap;
    uint32_t numExperts;
    float    eventThreshold;
};
static_assert(sizeof(FusedRootConstants) == 12 * 4, "Root constant size mismatch");

void FusedDispatch::Init(DX12Context*   ctx,
                          BufferManager* bufs,
                          const Config&  cfg)
{
    m_ctx  = ctx;
    m_bufs = bufs;
    m_cfg  = cfg;

    m_rootSig.Init(ctx->Device());
    m_pso.Init(ctx->Device(), m_rootSig.Get(),
               ShaderType::FusedAttentionForceMoE,
               "build/shaders");
}

void FusedDispatch::SetRootConstants(ID3D12GraphicsCommandList* cmdList) {
    FusedRootConstants rc{};
    rc.entityCount    = m_cfg.entityCount;
    rc.gridDimX       = m_cfg.gridDimX;
    rc.gridDimY       = m_cfg.gridDimY;
    rc.gridDimZ       = m_cfg.gridDimZ;
    rc.cellSize       = m_cfg.cellSize;
    rc.gridOriginX    = m_cfg.gridOriginX;
    rc.gridOriginY    = m_cfg.gridOriginY;
    rc.gridOriginZ    = m_cfg.gridOriginZ;
    rc.forceMax       = m_cfg.forceMax;
    rc.neighborCap    = m_cfg.neighborCap;
    rc.numExperts     = m_cfg.numExperts;
    rc.eventThreshold = m_cfg.eventThreshold;

    cmdList->SetComputeRoot32BitConstants(0, sizeof(rc) / 4, &rc, 0);
}

void FusedDispatch::BindBuffers(ID3D12GraphicsCommandList* cmdList) {
    auto& b = m_bufs->Buffers();
    // SRV inputs
    cmdList->SetComputeRootShaderResourceView(1,  b.entities  .Resource()->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(2,  b.position  .Resource()->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(3,  b.velocity  .Resource()->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(4,  b.signal    .Resource()->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(5,  b.axes      .Resource()->GetGPUVirtualAddress());
    // UAV outputs
    cmdList->SetComputeRootUnorderedAccessView(6,  b.force     .Resource()->GetGPUVirtualAddress());
    // Grid SRVs
    cmdList->SetComputeRootShaderResourceView(7,  b.gridOffsets.Resource()->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(8,  b.gridCounts .Resource()->GetGPUVirtualAddress());
    cmdList->SetComputeRootShaderResourceView(9,  b.gridIndices.Resource()->GetGPUVirtualAddress());
    // Remaining UAV outputs
    cmdList->SetComputeRootUnorderedAccessView(10, b.signalOut  .Resource()->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(11, b.events     .Resource()->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(12, b.eventParams.Resource()->GetGPUVirtualAddress());
}

void FusedDispatch::Dispatch(ID3D12GraphicsCommandList* cmdList) {
    cmdList->SetComputeRootSignature(m_rootSig.Get());
    cmdList->SetPipelineState(m_pso.Get());

    SetRootConstants(cmdList);
    BindBuffers(cmdList);

    uint32_t groups = (m_cfg.entityCount + 127u) / 128u;
    cmdList->Dispatch(groups, 1, 1);
}

} // namespace mx2lm
