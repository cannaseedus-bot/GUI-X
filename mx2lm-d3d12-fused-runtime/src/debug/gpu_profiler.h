#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <string>
#include <vector>
#include <cstdint>

using Microsoft::WRL::ComPtr;

namespace mx2lm {

struct ProfilerSample {
    std::string name;
    double      ms;
};

class GpuProfiler {
public:
    void Init(ID3D12Device* device, uint32_t maxSamples = 64);

    void BeginSample(ID3D12GraphicsCommandList* cmdList, const std::string& name);
    void EndSample  (ID3D12GraphicsCommandList* cmdList);

    // Resolve timestamps after ExecuteAndWait()
    void Resolve(ID3D12GraphicsCommandList* cmdList);

    // Read results from readback buffer (call after GPU flush)
    std::vector<ProfilerSample> GetResults();

    void Reset();

private:
    ComPtr<ID3D12QueryHeap> m_queryHeap;
    ComPtr<ID3D12Resource>  m_readback;
    uint64_t                m_gpuFreq   = 0;
    uint32_t                m_maxSamples = 0;
    uint32_t                m_sampleIdx  = 0;

    struct PendingSample {
        std::string name;
        uint32_t    startIdx;
        uint32_t    endIdx;
    };
    std::vector<PendingSample> m_pending;
};

} // namespace mx2lm
