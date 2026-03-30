#include "gpu_profiler.h"
#include "../dx12_context.h"
#include <stdexcept>

namespace mx2lm {

void GpuProfiler::Init(ID3D12Device* device, uint32_t maxSamples) {
    m_maxSamples = maxSamples;
    m_sampleIdx  = 0;

    D3D12_QUERY_HEAP_DESC qhd{};
    qhd.Type     = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qhd.Count    = maxSamples * 2;
    ThrowIfFailed(device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&m_queryHeap)),
                  "GpuProfiler::CreateQueryHeap");

    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = maxSamples * 2 * sizeof(uint64_t);
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags            = D3D12_RESOURCE_FLAG_NONE;

    ThrowIfFailed(device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&m_readback)), "GpuProfiler::CreateReadback");
}

void GpuProfiler::BeginSample(ID3D12GraphicsCommandList* cmdList,
                               const std::string&         name)
{
    if (m_sampleIdx >= m_maxSamples) return;

    uint32_t idx = m_sampleIdx * 2;
    cmdList->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, idx);

    m_pending.push_back({ name, idx, idx + 1 });
}

void GpuProfiler::EndSample(ID3D12GraphicsCommandList* cmdList) {
    if (m_pending.empty()) return;

    uint32_t endIdx = m_pending.back().endIdx;
    cmdList->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, endIdx);
    m_sampleIdx++;
}

void GpuProfiler::Resolve(ID3D12GraphicsCommandList* cmdList) {
    uint32_t count = m_sampleIdx * 2;
    if (count == 0) return;
    cmdList->ResolveQueryData(m_queryHeap.Get(),
                              D3D12_QUERY_TYPE_TIMESTAMP,
                              0, count,
                              m_readback.Get(), 0);
}

std::vector<ProfilerSample> GpuProfiler::GetResults() {
    if (m_gpuFreq == 0 || m_pending.empty()) return {};

    uint64_t* timestamps = nullptr;
    D3D12_RANGE readRange{ 0, m_sampleIdx * 2 * sizeof(uint64_t) };
    m_readback->Map(0, &readRange, (void**)&timestamps);

    std::vector<ProfilerSample> results;
    for (const auto& s : m_pending) {
        double ms = double(timestamps[s.endIdx] - timestamps[s.startIdx])
                  / double(m_gpuFreq) * 1000.0;
        results.push_back({ s.name, ms });
    }

    D3D12_RANGE empty{};
    m_readback->Unmap(0, &empty);
    return results;
}

void GpuProfiler::Reset() {
    m_sampleIdx = 0;
    m_pending.clear();
}

} // namespace mx2lm
