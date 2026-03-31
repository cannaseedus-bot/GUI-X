#include "pipeline_state.h"
#include "../dx12_context.h"
#include <fstream>
#include <vector>
#include <stdexcept>

namespace mx2lm {

std::string PipelineState::ShaderFileName(ShaderType type) {
    switch (type) {
        case ShaderType::FusedAttentionForceMoE: return "fused_attention_force_moe.cso";
        case ShaderType::GridBuildCount:         return "grid_build_count.cso";
        case ShaderType::GridBuildScan:          return "grid_build_scan.cso";
        case ShaderType::GridBuildScatter:       return "grid_build_scatter.cso";
        case ShaderType::PrefixSum:              return "prefix_sum.cso";
        case ShaderType::PrefixSumAddBlocks:     return "prefix_sum_add.cso";
        case ShaderType::DebugVisualize:         return "debug_visualize.cso";
        case ShaderType::AdamW:                  return "adamw.cso";
        case ShaderType::LoRAUpdate:             return "lora_update.cso";
        default: throw std::runtime_error("Unknown ShaderType");
    }
}

std::string PipelineState::EntryPoint(ShaderType) {
    return "CSMain";  // all shaders use CSMain as entry point
}

void PipelineState::Init(ID3D12Device*        device,
                          ID3D12RootSignature* rootSig,
                          ShaderType           type,
                          const std::string&   compiledShaderDir)
{
    std::string path = compiledShaderDir + "/" + ShaderFileName(type);

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Cannot open shader: " + path);

    size_t size = (size_t)file.tellg();
    file.seekg(0);
    std::vector<char> bytecode(size);
    file.read(bytecode.data(), size);

    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = rootSig;
    desc.CS.pShaderBytecode = bytecode.data();
    desc.CS.BytecodeLength  = size;

    ThrowIfFailed(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_pso)),
                  ("CreateComputePipelineState for " + path).c_str());
}

} // namespace mx2lm
