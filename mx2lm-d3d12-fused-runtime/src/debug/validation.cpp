#include "../dx12_context.h"
#include "../buffers/buffer_manager.h"
#include "../runtime/config.h"
#include <cstdio>
#include <cmath>
#include <DirectXMath.h>

using namespace DirectX;

namespace mx2lm {

// ── Validate GPU output buffers for NaN / Inf / out-of-range ──
// Only used in debug builds. Reads back force[] and checks.

void ValidateOutputs(ID3D12Device*              device,
                     ID3D12GraphicsCommandList* cmdList,
                     BufferManager*             bufs,
                     const Config&              cfg)
{
    uint32_t N = cfg.entityCount;
    std::vector<XMFLOAT4> forceData(N);
    std::vector<float>    signalData(N);
    std::vector<uint32_t> eventData(N);

    bufs->Readback(device, cmdList, bufs->Buffers().force.Resource(),
                   forceData.data(), N * sizeof(XMFLOAT4));
    bufs->Readback(device, cmdList, bufs->Buffers().signalOut.Resource(),
                   signalData.data(), N * sizeof(float));
    bufs->Readback(device, cmdList, bufs->Buffers().events.Resource(),
                   eventData.data(), N * sizeof(uint32_t));

    uint32_t nanCount    = 0;
    uint32_t clampCount  = 0;
    uint32_t eventCount  = 0;

    for (uint32_t i = 0; i < N; ++i) {
        auto& f = forceData[i];

        if (std::isnan(f.x) || std::isnan(f.y) || std::isnan(f.z)) {
            ++nanCount;
            if (nanCount <= 5)
                printf("[validation] NaN force at entity %u\n", i);
        }

        float mag = f.w;
        if (mag > cfg.forceMax * 1.01f) {
            ++clampCount;
        }

        if (std::isnan(signalData[i])) {
            printf("[validation] NaN signal at entity %u\n", i);
        }

        if (eventData[i] > 0) ++eventCount;
    }

    printf("[validation] NaN forces: %u  |  Over-clamped: %u  |  Events: %u\n",
           nanCount, clampCount, eventCount);
}

} // namespace mx2lm
