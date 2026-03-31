#pragma once
#include "../dx12_context.h"
#include "../buffers/buffer_manager.h"
#include "../pipeline/pipeline_state.h"
#include "../pipeline/root_signature.h"
#include <cstdint>

namespace mx2lm {

struct AdamWConfig {
    float    lr     = 1e-4f;
    float    beta1  = 0.9f;
    float    beta2  = 0.999f;
    float    eps    = 1e-8f;
    float    wd     = 0.01f;   // weight decay
    uint32_t paramCount = 0;
};

struct LoRAConfig {
    uint32_t rank   = 4;
    uint32_t dim    = 4;       // must match QKV projection dim
    float    scale  = 1.0f;    // α/r scaling
};

// ── AdamWGpu ──────────────────────────────────────────────────
// Manages weight/gradient/moment GPU buffers and dispatches
// AdamW update + LoRA update kernels.

class AdamWGpu {
public:
    void Init(DX12Context*       ctx,
              const AdamWConfig& adamCfg,
              const LoRAConfig&  loraCfg,
              const std::string& shaderDir);

    // Dispatch AdamW weight update (call after backward pass)
    void StepWeights(ID3D12GraphicsCommandList* cmdList, uint32_t step);

    // Dispatch LoRA decay pass
    void StepLoRA(ID3D12GraphicsCommandList* cmdList, uint32_t step);

    // GPU buffer accessors for binding to fused kernel
    ID3D12Resource* WeightsBuf()  const;
    ID3D12Resource* GradsBuf()    const;
    ID3D12Resource* LoRAaBuf()    const;
    ID3D12Resource* LoRAbBuf()    const;

private:
    DX12Context*  m_ctx = nullptr;
    AdamWConfig   m_adamCfg;
    LoRAConfig    m_loraCfg;

    // Parameter buffers
    struct Bufs {
        using F = StructuredBuffer<float>;
        F weights, grads, m_mom, v_var;
        F lora_a,  lora_b;
    } m_bufs;

    RootSignature m_rootSig;
    PipelineState m_psoAdamW;
    PipelineState m_psoLoRA;

    void InitBuffers(ID3D12Device* device);
    void InitRootSig(ID3D12Device* device);
};

} // namespace mx2lm
