#pragma once
#include "../dx12_context.h"
#include "../buffers/buffer_manager.h"
#include "../pipeline/pipeline_state.h"
#include "../pipeline/root_signature.h"
#include "../runtime/config.h"

namespace mx2lm {

class GridDispatch {
public:
    void Init(DX12Context* ctx, BufferManager* bufs, const Config& cfg);

    // Execute all three grid-build passes (count → scan → scatter)
    // with UAV barriers between each.
    void BuildGrid(ID3D12GraphicsCommandList* cmdList);

private:
    DX12Context*   m_ctx  = nullptr;
    BufferManager* m_bufs = nullptr;
    Config         m_cfg;

    RootSignature  m_rootSig;
    PipelineState  m_psoCount;
    PipelineState  m_psoScan;
    PipelineState  m_psoScatter;

    void ClearGridBuffers(ID3D12GraphicsCommandList* cmdList);
    void SetGridRootConstants(ID3D12GraphicsCommandList* cmdList, uint32_t passIndex);
};

} // namespace mx2lm
