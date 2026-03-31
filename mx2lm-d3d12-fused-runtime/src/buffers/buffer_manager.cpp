#include "buffer_manager.h"
#include "../dx12_context.h"
#include <stdexcept>

namespace mx2lm {

void BufferManager::Init(ID3D12Device* device, const Config& cfg) {
    uint32_t N    = cfg.entityCount;
    uint32_t G    = cfg.gridCellCount();
    uint32_t axes = N * 3u;  // 3 float4 rows per entity

    // ── Allocate entity buffers ───────────────────────────────
    m_buf.entities  .Allocate(device, N, D3D12_RESOURCE_FLAG_NONE);
    m_buf.position  .Allocate(device, N, D3D12_RESOURCE_FLAG_NONE);
    m_buf.velocity  .Allocate(device, N, D3D12_RESOURCE_FLAG_NONE);
    m_buf.signal    .Allocate(device, N, D3D12_RESOURCE_FLAG_NONE);
    m_buf.axes      .Allocate(device, axes, D3D12_RESOURCE_FLAG_NONE);

    // ── Allocate output buffers (UAV) ─────────────────────────
    m_buf.force      .Allocate(device, N);
    m_buf.signalOut  .Allocate(device, N);
    m_buf.events     .Allocate(device, N);
    m_buf.eventParams.Allocate(device, N);

    // ── Allocate grid buffers ─────────────────────────────────
    m_buf.gridOffsets  .Allocate(device, G);
    m_buf.gridCounts   .Allocate(device, G);
    m_buf.gridIndices  .Allocate(device, N);  // worst case: all in one cell
    m_buf.scatterCursor.Allocate(device, G);
}

void BufferManager::Upload(ID3D12GraphicsCommandList* cmdList,
                           ID3D12Resource*            dst,
                           const void*                src,
                           uint64_t                   byteSize)
{
    // Create a temporary upload heap buffer
    // (in production this should be pooled)
    ComPtr<ID3D12Resource> staging;
    ID3D12Device* device = nullptr;
    dst->GetDevice(IID_PPV_ARGS(&device));

    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = byteSize;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags            = D3D12_RESOURCE_FLAG_NONE;

    ThrowIfFailed(device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&staging)), "Upload::CreateStagingBuffer");

    // Copy CPU → staging
    void* mapped = nullptr;
    staging->Map(0, nullptr, &mapped);
    memcpy(mapped, src, byteSize);
    D3D12_RANGE written{0, byteSize};
    staging->Unmap(0, &written);

    // Transition dst COMMON → COPY_DEST
    Transition(cmdList, dst,
               D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_COPY_DEST);

    cmdList->CopyBufferRegion(dst, 0, staging.Get(), 0, byteSize);

    // Transition dst COPY_DEST → COMMON (or SRV/UAV depending on usage)
    Transition(cmdList, dst,
               D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_COMMON);
}

void BufferManager::Readback(ID3D12Device*              device,
                              ID3D12GraphicsCommandList* cmdList,
                              ID3D12Resource*            src,
                              void*                      dst,
                              uint64_t                   byteSize)
{
    ComPtr<ID3D12Resource> readback;

    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = byteSize;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags            = D3D12_RESOURCE_FLAG_NONE;

    ThrowIfFailed(device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&readback)), "Readback::CreateReadbackBuffer");

    Transition(cmdList, src,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);

    cmdList->CopyBufferRegion(readback.Get(), 0, src, 0, byteSize);

    Transition(cmdList, src,
               D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Note: caller must flush command list before mapping readback
    void* mapped = nullptr;
    readback->Map(0, nullptr, &mapped);
    memcpy(dst, mapped, byteSize);
    readback->Unmap(0, nullptr);
}

void BufferManager::Transition(ID3D12GraphicsCommandList* cmdList,
                                ID3D12Resource*            res,
                                D3D12_RESOURCE_STATES      before,
                                D3D12_RESOURCE_STATES      after)
{
    D3D12_RESOURCE_BARRIER b{};
    b.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource   = res;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter  = after;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &b);
}

void BufferManager::UAVBarrier(ID3D12GraphicsCommandList* cmdList,
                                ID3D12Resource*            res)
{
    D3D12_RESOURCE_BARRIER b{};
    b.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = res;
    cmdList->ResourceBarrier(1, &b);
}

} // namespace mx2lm
