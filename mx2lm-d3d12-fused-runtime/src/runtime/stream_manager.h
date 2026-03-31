#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <unordered_map>
#include <deque>
#include <cstdint>
#include <cstddef>

using Microsoft::WRL::ComPtr;

namespace mx2lm {

// ── StreamPage ────────────────────────────────────────────────
// One 4 MB (configurable) page of VRAM.

constexpr uint64_t STREAM_PAGE_SIZE = 4ull * 1024 * 1024;   // 4 MB
constexpr uint32_t VRAM_PAGE_BUDGET = 2048;                  // ~8 GB max

struct StreamPage {
    uint64_t               key;       // content-addressed: byte offset in blob
    ComPtr<ID3D12Resource> resource;  // GPU DEFAULT heap resource
    bool                   dirty = false;
};

// ── StreamManager ─────────────────────────────────────────────
// Elastic VRAM tier:
//   - keeps hot pages in GPU DEFAULT heap
//   - evicts LRU pages when over budget
//   - zero-copy upload from file-mapped or RAM-cached regions

class StreamManager {
public:
    void Init(ID3D12Device* device, uint32_t maxPages = VRAM_PAGE_BUDGET);

    // Upload a raw memory region → GPU page.
    // Returns the GPU-side resource (valid until eviction).
    ID3D12Resource* StreamIn(ID3D12GraphicsCommandList* cmdList,
                             uint64_t                   key,
                             const void*                srcData,
                             uint64_t                   byteSize);

    // Check if a page is resident
    bool IsResident(uint64_t key) const;

    // Explicitly evict a page (frees VRAM)
    void Evict(uint64_t key);

    // Evict LRU pages until under budget
    void Trim();

    uint32_t ResidentCount() const { return (uint32_t)m_pages.size(); }

private:
    ID3D12Device*                              m_device  = nullptr;
    uint32_t                                   m_maxPages = 0;

    std::unordered_map<uint64_t, StreamPage>   m_pages;
    std::deque<uint64_t>                       m_lruQueue;  // front = oldest

    StreamPage AllocPage(uint64_t key, uint64_t byteSize);
    void       UploadToPage(ID3D12GraphicsCommandList* cmdList,
                            StreamPage& page,
                            const void* src,
                            uint64_t    byteSize);
    void       EvictOldest();
};

} // namespace mx2lm
