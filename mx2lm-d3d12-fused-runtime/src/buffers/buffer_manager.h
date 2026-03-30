#pragma once
#include "structured_buffer.h"
#include "../runtime/config.h"
#include <d3d12.h>
#include <DirectXMath.h>
#include <memory>

namespace mx2lm {

// ── GpuBuffers ────────────────────────────────────────────────
// All GPU buffers for the simulation, matching memory_layout.md

struct GpuBuffers {
    // Entity state (SRV inputs)
    StructuredBuffer<uint32_t>            entities;
    StructuredBuffer<DirectX::XMFLOAT4>   position;
    StructuredBuffer<DirectX::XMFLOAT4>   velocity;
    StructuredBuffer<float>               signal;
    StructuredBuffer<DirectX::XMFLOAT4>   axes;       // 3 float4 per entity

    // Output buffers (UAV)
    StructuredBuffer<DirectX::XMFLOAT4>   force;
    StructuredBuffer<float>               signalOut;
    StructuredBuffer<uint32_t>            events;
    StructuredBuffer<DirectX::XMFLOAT4>   eventParams;

    // Grid buffers
    StructuredBuffer<uint32_t>            gridOffsets;
    StructuredBuffer<uint32_t>            gridCounts;
    StructuredBuffer<uint32_t>            gridIndices;
    StructuredBuffer<uint32_t>            scatterCursor;  // temp, grid_build only
};

// ── BufferManager ─────────────────────────────────────────────

class BufferManager {
public:
    BufferManager() = default;

    void Init(ID3D12Device* device, const Config& cfg);

    // Upload CPU data to GPU via staging buffer
    void Upload(ID3D12GraphicsCommandList* cmdList,
                ID3D12Resource*            dstBuffer,
                const void*                srcData,
                uint64_t                   byteSize);

    // Readback GPU buffer to CPU (debug only)
    void Readback(ID3D12Device*             device,
                  ID3D12GraphicsCommandList* cmdList,
                  ID3D12Resource*            srcBuffer,
                  void*                      dstData,
                  uint64_t                   byteSize);

    GpuBuffers& Buffers() { return m_buf; }

    // Transition a buffer between resource states
    static void Transition(ID3D12GraphicsCommandList* cmdList,
                           ID3D12Resource*            res,
                           D3D12_RESOURCE_STATES      before,
                           D3D12_RESOURCE_STATES      after);

    static void UAVBarrier(ID3D12GraphicsCommandList* cmdList,
                           ID3D12Resource*            res);

private:
    GpuBuffers m_buf;
};

} // namespace mx2lm
