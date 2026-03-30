// ============================================================
//  prefix_sum.hlsl
//  MX2LM D3D12 Fused Runtime
//
//  Parallel exclusive prefix sum (scan) using shared memory.
//  Used by grid_build when gridCellCount > 65536.
//
//  Algorithm: Blelloch work-efficient scan
//  SM 6.0 — uses wave intrinsics (documented exception to
//           deterministic_rules.md § no-wave-intrinsics)
// ============================================================

#include "common.hlsli"

cbuffer RootConstants : register(b0)
{
    uint elementCount;
    uint pad0;
    uint pad1;
    uint pad2;
}

RWStructuredBuffer<uint> data    : register(u0);
RWStructuredBuffer<uint> blockSums : register(u1);  // partial sums per block

groupshared uint sharedData[512];

[numthreads(256, 1, 1)]
void CSScan(uint3 gid  : SV_GroupID,
            uint3 gtid : SV_GroupThreadID,
            uint3 tid  : SV_DispatchThreadID)
{
    uint localIdx  = gtid.x;
    uint globalIdx = tid.x * 2u;

    // Load two elements per thread
    sharedData[localIdx * 2u]      = (globalIdx     < elementCount) ? data[globalIdx]     : 0u;
    sharedData[localIdx * 2u + 1u] = (globalIdx + 1u < elementCount) ? data[globalIdx + 1u] : 0u;
    GroupMemoryBarrierWithGroupSync();

    // Up-sweep (reduce)
    uint offset = 1u;
    for (uint d = 256u; d > 0u; d >>= 1)
    {
        GroupMemoryBarrierWithGroupSync();
        if (localIdx < d)
        {
            uint ai = offset * (localIdx * 2u + 1u) - 1u;
            uint bi = offset * (localIdx * 2u + 2u) - 1u;
            sharedData[bi] += sharedData[ai];
        }
        offset <<= 1;
    }

    // Save total and clear last element
    if (localIdx == 0u)
    {
        blockSums[gid.x] = sharedData[511];
        sharedData[511]  = 0u;
    }

    // Down-sweep
    for (uint d = 1u; d < 512u; d <<= 1)
    {
        offset >>= 1;
        GroupMemoryBarrierWithGroupSync();
        if (localIdx < d)
        {
            uint ai  = offset * (localIdx * 2u + 1u) - 1u;
            uint bi  = offset * (localIdx * 2u + 2u) - 1u;
            uint tmp    = sharedData[ai];
            sharedData[ai] = sharedData[bi];
            sharedData[bi] += tmp;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Write back
    if (globalIdx     < elementCount) data[globalIdx]      = sharedData[localIdx * 2u];
    if (globalIdx + 1u < elementCount) data[globalIdx + 1u] = sharedData[localIdx * 2u + 1u];
}

// ── Add block sums pass ───────────────────────────────────────
[numthreads(512, 1, 1)]
void CSAddBlockSums(uint3 gid : SV_GroupID, uint3 tid : SV_DispatchThreadID)
{
    if (tid.x < elementCount)
        data[tid.x] += blockSums[gid.x];
}
