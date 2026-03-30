#pragma once
#include <d3d12.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

namespace mx2lm {

// Root signature layout — must match memory_layout.md exactly
//
// Slot  Type   Register   Buffer
// ───────────────────────────────────────────
//  0    CBV    b0         Root constants (12 × uint/float)
//  1    SRV    t0         entities
//  2    SRV    t1         position
//  3    SRV    t2         velocity
//  4    SRV    t3         signalIn
//  5    SRV    t4         axes
//  6    UAV    u5         force
//  7    SRV    t6         gridOffsets
//  8    SRV    t7         gridCounts
//  9    SRV    t8         gridIndices
// 10    UAV    u3         signalOut
// 11    UAV    u9         events
// 12    UAV    u10        eventParams

class RootSignature {
public:
    void Init(ID3D12Device* device);

    ID3D12RootSignature* Get() const { return m_rootSig.Get(); }

private:
    ComPtr<ID3D12RootSignature> m_rootSig;
};

} // namespace mx2lm
