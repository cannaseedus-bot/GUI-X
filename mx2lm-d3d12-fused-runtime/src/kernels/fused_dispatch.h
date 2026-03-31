#pragma once
#include "../dx12_context.h"
#include "../buffers/buffer_manager.h"
#include "../pipeline/pipeline_state.h"
#include "../pipeline/root_signature.h"
#include "../runtime/config.h"

namespace mx2lm {

class FusedDispatch {
public:
    void Init(DX12Context* ctx, BufferManager* bufs, const Config& cfg);

    // Dispatch the fused attention+force+MoE kernel
    // cmdList must be open and have root signature set
    void Dispatch(ID3D12GraphicsCommandList* cmdList);

private:
    DX12Context*   m_ctx  = nullptr;
    BufferManager* m_bufs = nullptr;
    Config         m_cfg;

    RootSignature  m_rootSig;
    PipelineState  m_pso;

    void BindBuffers(ID3D12GraphicsCommandList* cmdList);
    void SetRootConstants(ID3D12GraphicsCommandList* cmdList);
};

} // namespace mx2lm
