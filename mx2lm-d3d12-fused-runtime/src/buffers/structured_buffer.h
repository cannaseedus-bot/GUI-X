#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>
#include <stdexcept>

using Microsoft::WRL::ComPtr;

namespace mx2lm {

// ── StructuredBuffer ─────────────────────────────────────────
// Generic RAII wrapper for a D3D12 structured buffer in DEFAULT heap.
// Provides SRV and UAV descriptor creation helpers.
//
template<typename T>
class StructuredBuffer {
public:
    StructuredBuffer() = default;

    void Allocate(ID3D12Device* device, uint32_t elementCount,
                  D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
    {
        m_elementCount = elementCount;
        m_elementSize  = sizeof(T);

        uint64_t byteSize = (uint64_t)elementCount * sizeof(T);
        // Align to 16 bytes
        byteSize = (byteSize + 15ull) & ~15ull;

        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;

        D3D12_RESOURCE_DESC rd{};
        rd.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width              = byteSize;
        rd.Height             = 1;
        rd.DepthOrArraySize   = 1;
        rd.MipLevels          = 1;
        rd.SampleDesc.Count   = 1;
        rd.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags              = flags;

        HRESULT hr = device->CreateCommittedResource(
            &hp,
            D3D12_HEAP_FLAG_NONE,
            &rd,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&m_buffer));

        if (FAILED(hr))
            throw std::runtime_error("StructuredBuffer::Allocate failed");
    }

    void CreateSRV(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE handle) const {
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.ViewDimension              = D3D12_SRV_DIMENSION_BUFFER;
        d.Shader4ComponentMapping    = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Buffer.NumElements         = m_elementCount;
        d.Buffer.StructureByteStride = m_elementSize;
        d.Buffer.Flags               = D3D12_BUFFER_SRV_FLAG_NONE;
        device->CreateShaderResourceView(m_buffer.Get(), &d, handle);
    }

    void CreateUAV(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE handle) const {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.ViewDimension              = D3D12_UAV_DIMENSION_BUFFER;
        d.Buffer.NumElements         = m_elementCount;
        d.Buffer.StructureByteStride = m_elementSize;
        d.Buffer.Flags               = D3D12_BUFFER_UAV_FLAG_NONE;
        device->CreateUnorderedAccessView(m_buffer.Get(), nullptr, &d, handle);
    }

    ID3D12Resource* Resource()     const { return m_buffer.Get(); }
    uint32_t        ElementCount() const { return m_elementCount; }
    uint32_t        ByteSize()     const { return m_elementCount * m_elementSize; }

private:
    ComPtr<ID3D12Resource> m_buffer;
    uint32_t               m_elementCount = 0;
    uint32_t               m_elementSize  = 0;
};

} // namespace mx2lm
