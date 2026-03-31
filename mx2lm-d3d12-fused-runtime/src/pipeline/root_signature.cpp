#include "root_signature.h"
#include "../dx12_context.h"
#include <d3dcompiler.h>
#include <stdexcept>
#include <vector>

namespace mx2lm {

void RootSignature::Init(ID3D12Device* device) {
    // 13 root parameters (see root_signature.h)
    std::vector<D3D12_ROOT_PARAMETER1> params(13);

    auto MakeCBV = [](UINT reg, D3D12_SHADER_VISIBILITY vis = D3D12_SHADER_VISIBILITY_ALL) {
        D3D12_ROOT_PARAMETER1 p{};
        p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        p.Descriptor.ShaderRegister = reg;
        p.ShaderVisibility = vis;
        return p;
    };

    auto MakeSRV = [](UINT reg) {
        D3D12_ROOT_PARAMETER1 p{};
        p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
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

    params[0]  = MakeCBV(0);   // b0  root constants
    params[1]  = MakeSRV(0);   // t0  entities
    params[2]  = MakeSRV(1);   // t1  position
    params[3]  = MakeSRV(2);   // t2  velocity
    params[4]  = MakeSRV(3);   // t3  signalIn
    params[5]  = MakeSRV(4);   // t4  axes
    params[6]  = MakeUAV(5);   // u5  force
    params[7]  = MakeSRV(6);   // t6  gridOffsets
    params[8]  = MakeSRV(7);   // t7  gridCounts
    params[9]  = MakeSRV(8);   // t8  gridIndices
    params[10] = MakeUAV(3);   // u3  signalOut
    params[11] = MakeUAV(9);   // u9  events
    params[12] = MakeUAV(10);  // u10 eventParams

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    desc.Desc_1_1.NumParameters = (UINT)params.size();
    desc.Desc_1_1.pParameters   = params.data();
    desc.Desc_1_1.Flags         = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> serialized, error;
    HRESULT hr = D3D12SerializeVersionedRootSignature(&desc,
                                                      &serialized,
                                                      &error);
    if (FAILED(hr)) {
        std::string msg = "SerializeRootSignature failed";
        if (error)
            msg += ": " + std::string((char*)error->GetBufferPointer());
        throw std::runtime_error(msg);
    }

    ThrowIfFailed(device->CreateRootSignature(
        0,
        serialized->GetBufferPointer(),
        serialized->GetBufferSize(),
        IID_PPV_ARGS(&m_rootSig)),
        "CreateRootSignature");
}

} // namespace mx2lm
