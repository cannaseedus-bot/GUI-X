#include "adamw_gpu.h"
#include "../dx12_context.h"
#include <cstring>

namespace mx2lm {

// ── Root constant layout for AdamW cbuffer ────────────────────
struct AdamWRootConst {
    float    lr;
    float    beta1;
    float    beta2;
    float    eps;
    float    wd;
    uint32_t step;
    uint32_t paramCount;
    uint32_t pad;
};
static_assert(sizeof(AdamWRootConst) == 32, "AdamW root const size");

// ── Init ──────────────────────────────────────────────────────

void AdamWGpu::Init(DX12Context*       ctx,
                     const AdamWConfig& adamCfg,
                     const LoRAConfig&  loraCfg,
                     const std::string& shaderDir)
{
    m_ctx     = ctx;
    m_adamCfg = adamCfg;
    m_loraCfg = loraCfg;

    InitBuffers(ctx->Device());
    InitRootSig(ctx->Device());

    m_psoAdamW.Init(ctx->Device(), m_rootSig.Get(),
                    ShaderType::AdamW, shaderDir);
    m_psoLoRA .Init(ctx->Device(), m_rootSig.Get(),
                    ShaderType::LoRAUpdate, shaderDir);
}

void AdamWGpu::InitBuffers(ID3D12Device* device) {
    uint32_t N     = m_adamCfg.paramCount;
    uint32_t loraN = m_loraCfg.rank * N;   // rank × params for each factor

    m_bufs.weights.Allocate(device, N);
    m_bufs.grads  .Allocate(device, N);
    m_bufs.m_mom  .Allocate(device, N);
    m_bufs.v_var  .Allocate(device, N);
    m_bufs.lora_a .Allocate(device, loraN);
    m_bufs.lora_b .Allocate(device, loraN);
}

void AdamWGpu::InitRootSig(ID3D12Device* device) {
    // Simple root sig: CBV b0 + 6 UAVs (u0..u5)
    std::vector<D3D12_ROOT_PARAMETER1> params(7);

    auto MakeCBV = [](UINT reg) {
        D3D12_ROOT_PARAMETER1 p{};
        p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        p.Descriptor.ShaderRegister = reg;
        p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        return p;
    };
    auto MakeUAV = [](UINT reg) {
        D3D12_ROOT_PARAMETER1 p{};
        p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        p.Descriptor.ShaderRegister = reg;
        p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        return p;
    };

    params[0] = MakeCBV(0);   // b0 AdamWParams
    params[1] = MakeUAV(0);   // u0 weights
    params[2] = MakeUAV(1);   // u1 grads
    params[3] = MakeUAV(2);   // u2 m (1st moment)
    params[4] = MakeUAV(3);   // u3 v (2nd moment)
    params[5] = MakeUAV(4);   // u4 lora_A
    params[6] = MakeUAV(5);   // u5 lora_B

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    desc.Desc_1_1.NumParameters = (UINT)params.size();
    desc.Desc_1_1.pParameters   = params.data();
    desc.Desc_1_1.Flags         = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> serialized, error;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&desc, &serialized, &error);
    if (FAILED(hr)) {
        std::string msg = "AdamW SerializeRootSignature";
        if (error) msg += ": " + std::string((char*)error->GetBufferPointer());
        throw std::runtime_error(msg);
    }

    ThrowIfFailed(device->CreateRootSignature(
        0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
        IID_PPV_ARGS(&m_rootSig.Get())),
        "AdamW CreateRootSignature");
}

// ── Dispatch helpers ──────────────────────────────────────────

static void BindAdamWBuffers(ID3D12GraphicsCommandList* cmd,
                              AdamWGpu::Bufs&            b,
                              const AdamWRootConst&      rc)
{
    cmd->SetComputeRoot32BitConstants(0, sizeof(rc) / 4, &rc, 0);
    cmd->SetComputeRootUnorderedAccessView(1, b.weights.Resource()->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(2, b.grads  .Resource()->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(3, b.m_mom  .Resource()->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(4, b.v_var  .Resource()->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(5, b.lora_a .Resource()->GetGPUVirtualAddress());
    cmd->SetComputeRootUnorderedAccessView(6, b.lora_b .Resource()->GetGPUVirtualAddress());
}

void AdamWGpu::StepWeights(ID3D12GraphicsCommandList* cmdList, uint32_t step) {
    AdamWRootConst rc{};
    rc.lr         = m_adamCfg.lr;
    rc.beta1      = m_adamCfg.beta1;
    rc.beta2      = m_adamCfg.beta2;
    rc.eps        = m_adamCfg.eps;
    rc.wd         = m_adamCfg.wd;
    rc.step       = step;
    rc.paramCount = m_adamCfg.paramCount;

    cmdList->SetComputeRootSignature(m_rootSig.Get());
    cmdList->SetPipelineState(m_psoAdamW.Get());
    BindAdamWBuffers(cmdList, m_bufs, rc);

    UINT groups = (m_adamCfg.paramCount + 127u) / 128u;
    cmdList->Dispatch(groups, 1, 1);
}

void AdamWGpu::StepLoRA(ID3D12GraphicsCommandList* cmdList, uint32_t step) {
    AdamWRootConst rc{};
    rc.lr         = m_adamCfg.lr;
    rc.wd         = m_adamCfg.wd;
    rc.step       = step;
    rc.paramCount = m_loraCfg.rank * m_adamCfg.paramCount;

    cmdList->SetComputeRootSignature(m_rootSig.Get());
    cmdList->SetPipelineState(m_psoLoRA.Get());
    BindAdamWBuffers(cmdList, m_bufs, rc);

    UINT groups = (rc.paramCount + 127u) / 128u;
    cmdList->Dispatch(groups, 1, 1);
}

// ── Buffer accessors ──────────────────────────────────────────
ID3D12Resource* AdamWGpu::WeightsBuf() const { return m_bufs.weights.Resource(); }
ID3D12Resource* AdamWGpu::GradsBuf()   const { return m_bufs.grads  .Resource(); }
ID3D12Resource* AdamWGpu::LoRAaBuf()   const { return m_bufs.lora_a .Resource(); }
ID3D12Resource* AdamWGpu::LoRAbBuf()   const { return m_bufs.lora_b .Resource(); }

} // namespace mx2lm
