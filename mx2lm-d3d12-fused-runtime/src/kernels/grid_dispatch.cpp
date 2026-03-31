#include "grid_dispatch.h"

namespace mx2lm {

struct GridRootConstants {
    uint32_t entityCount;
    uint32_t gridDimX;
    uint32_t gridDimY;
    uint32_t gridDimZ;
    float    cellSize;
    float    gridOriginX;
    float    gridOriginY;
    float    gridOriginZ;
    uint32_t passIndex;
    uint32_t gridCellCount;
    uint32_t pad0;
    uint32_t pad1;
};

void GridDispatch::Init(DX12Context*   ctx,
                         BufferManager* bufs,
                         const Config&  cfg)
{
    m_ctx  = ctx;
    m_bufs = bufs;
    m_cfg  = cfg;

    m_rootSig.Init(ctx->Device());
    m_psoCount  .Init(ctx->Device(), m_rootSig.Get(), ShaderType::GridBuildCount,   "build/shaders");
    m_psoScan   .Init(ctx->Device(), m_rootSig.Get(), ShaderType::GridBuildScan,    "build/shaders");
    m_psoScatter.Init(ctx->Device(), m_rootSig.Get(), ShaderType::GridBuildScatter, "build/shaders");
}

void GridDispatch::SetGridRootConstants(ID3D12GraphicsCommandList* cmdList,
                                         uint32_t                   passIndex)
{
    GridRootConstants rc{};
    rc.entityCount   = m_cfg.entityCount;
    rc.gridDimX      = m_cfg.gridDimX;
    rc.gridDimY      = m_cfg.gridDimY;
    rc.gridDimZ      = m_cfg.gridDimZ;
    rc.cellSize      = m_cfg.cellSize;
    rc.gridOriginX   = m_cfg.gridOriginX;
    rc.gridOriginY   = m_cfg.gridOriginY;
    rc.gridOriginZ   = m_cfg.gridOriginZ;
    rc.passIndex     = passIndex;
    rc.gridCellCount = m_cfg.gridCellCount();
    cmdList->SetComputeRoot32BitConstants(0, sizeof(rc) / 4, &rc, 0);
}

void GridDispatch::ClearGridBuffers(ID3D12GraphicsCommandList* cmdList) {
    // Clear gridCounts and scatterCursor to 0
    auto& b = m_bufs->Buffers();
    UINT clearVal[4] = {0, 0, 0, 0};

    // We need descriptor handles for ClearUnorderedAccessViewUint.
    // For simplicity in MVP, we zero-fill via an upload instead.
    // (Full implementation: use ClearUnorderedAccessViewUint with heap descriptors)
    (void)b; (void)clearVal; (void)cmdList;
    // TODO: implement descriptor-heap-based UAV clear
}

void GridDispatch::BuildGrid(ID3D12GraphicsCommandList* cmdList) {
    auto& b = m_bufs->Buffers();
    cmdList->SetComputeRootSignature(m_rootSig.Get());

    // ── Pass 0: Count ─────────────────────────────────────────
    cmdList->SetPipelineState(m_psoCount.Get());
    SetGridRootConstants(cmdList, 0);
    cmdList->SetComputeRootShaderResourceView(2, b.position.Resource()->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(6, b.gridCounts.Resource()->GetGPUVirtualAddress());
    uint32_t entityGroups = (m_cfg.entityCount + 127u) / 128u;
    cmdList->Dispatch(entityGroups, 1, 1);

    BufferManager::UAVBarrier(cmdList, b.gridCounts.Resource());

    // ── Pass 1: Prefix Sum ────────────────────────────────────
    cmdList->SetPipelineState(m_psoScan.Get());
    SetGridRootConstants(cmdList, 1);
    cmdList->SetComputeRootUnorderedAccessView(6, b.gridCounts  .Resource()->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(10, b.gridOffsets.Resource()->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(11, b.scatterCursor.Resource()->GetGPUVirtualAddress());
    cmdList->Dispatch(1, 1, 1);

    BufferManager::UAVBarrier(cmdList, b.gridOffsets.Resource());
    BufferManager::UAVBarrier(cmdList, b.scatterCursor.Resource());

    // ── Pass 2: Scatter ───────────────────────────────────────
    cmdList->SetPipelineState(m_psoScatter.Get());
    SetGridRootConstants(cmdList, 2);
    cmdList->SetComputeRootShaderResourceView(2,   b.position    .Resource()->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(10,  b.gridIndices .Resource()->GetGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(11,  b.scatterCursor.Resource()->GetGPUVirtualAddress());
    cmdList->Dispatch(entityGroups, 1, 1);

    BufferManager::UAVBarrier(cmdList, b.gridIndices.Resource());
}

} // namespace mx2lm
