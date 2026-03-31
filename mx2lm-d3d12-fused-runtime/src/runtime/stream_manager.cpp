#include "stream_manager.h"
#include "../dx12_context.h"
#include <stdexcept>
#include <algorithm>

namespace mx2lm {

void StreamManager::Init(ID3D12Device* device, uint32_t maxPages) {
    m_device   = device;
    m_maxPages = maxPages;
}

// ── Alloc a new DEFAULT-heap page ─────────────────────────────
StreamPage StreamManager::AllocPage(uint64_t key, uint64_t byteSize) {
    uint64_t alignedSize = (byteSize + 15ull) & ~15ull;

    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = alignedSize;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    StreamPage page;
    page.key = key;
    ThrowIfFailed(m_device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_COMMON, nullptr,
        IID_PPV_ARGS(&page.resource)), "StreamManager::AllocPage");

    return page;
}

// ── Upload src → DEFAULT heap via inline staging ──────────────
void StreamManager::UploadToPage(ID3D12GraphicsCommandList* cmdList,
                                  StreamPage&                page,
                                  const void*                src,
                                  uint64_t                   byteSize)
{
    uint64_t alignedSize = (byteSize + 15ull) & ~15ull;

    // Staging buffer in UPLOAD heap
    ComPtr<ID3D12Resource> staging;
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = alignedSize;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags            = D3D12_RESOURCE_FLAG_NONE;

    ThrowIfFailed(m_device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&staging)), "StreamManager::UploadToPage staging");

    // CPU → staging (zero-copy when src is file-mapped memory)
    void* mapped = nullptr;
    staging->Map(0, nullptr, &mapped);
    memcpy(mapped, src, byteSize);
    D3D12_RANGE written{ 0, byteSize };
    staging->Unmap(0, &written);

    // Transition page COMMON → COPY_DEST
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource   = page.resource.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &barrier);

    cmdList->CopyBufferRegion(page.resource.Get(), 0, staging.Get(), 0, byteSize);

    // Transition COPY_DEST → COMMON (ready for SRV/UAV bind)
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COMMON;
    cmdList->ResourceBarrier(1, &barrier);
}

// ── StreamIn ──────────────────────────────────────────────────
ID3D12Resource* StreamManager::StreamIn(ID3D12GraphicsCommandList* cmdList,
                                         uint64_t                   key,
                                         const void*                srcData,
                                         uint64_t                   byteSize)
{
    // Already resident?
    auto it = m_pages.find(key);
    if (it != m_pages.end()) {
        // Refresh LRU
        auto lruIt = std::find(m_lruQueue.begin(), m_lruQueue.end(), key);
        if (lruIt != m_lruQueue.end()) m_lruQueue.erase(lruIt);
        m_lruQueue.push_back(key);
        return it->second.resource.Get();
    }

    // Evict if over budget
    while (m_pages.size() >= m_maxPages)
        EvictOldest();

    StreamPage page = AllocPage(key, byteSize);
    UploadToPage(cmdList, page, srcData, byteSize);

    ID3D12Resource* res = page.resource.Get();
    m_pages[key]        = std::move(page);
    m_lruQueue.push_back(key);

    return res;
}

// ── IsResident ────────────────────────────────────────────────
bool StreamManager::IsResident(uint64_t key) const {
    return m_pages.count(key) > 0;
}

// ── Evict ─────────────────────────────────────────────────────
void StreamManager::Evict(uint64_t key) {
    m_pages.erase(key);
    auto it = std::find(m_lruQueue.begin(), m_lruQueue.end(), key);
    if (it != m_lruQueue.end()) m_lruQueue.erase(it);
}

void StreamManager::EvictOldest() {
    if (m_lruQueue.empty()) return;
    uint64_t oldest = m_lruQueue.front();
    m_lruQueue.pop_front();
    m_pages.erase(oldest);
}

void StreamManager::Trim() {
    while (m_pages.size() > m_maxPages)
        EvictOldest();
}

} // namespace mx2lm
