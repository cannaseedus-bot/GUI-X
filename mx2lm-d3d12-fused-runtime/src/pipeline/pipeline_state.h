#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <string>

using Microsoft::WRL::ComPtr;

namespace mx2lm {

enum class ShaderType {
    FusedAttentionForceMoE,
    GridBuildCount,
    GridBuildScan,
    GridBuildScatter,
    PrefixSum,
    PrefixSumAddBlocks,
    DebugVisualize,
    AdamW,
    LoRAUpdate,
};

class PipelineState {
public:
    void Init(ID3D12Device*        device,
              ID3D12RootSignature* rootSig,
              ShaderType           type,
              const std::string&   compiledShaderDir);

    ID3D12PipelineState* Get() const { return m_pso.Get(); }

private:
    ComPtr<ID3D12PipelineState> m_pso;

    static std::string ShaderFileName(ShaderType type);
    static std::string EntryPoint(ShaderType type);
};

} // namespace mx2lm
