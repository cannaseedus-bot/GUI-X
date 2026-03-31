// ============================================================
//  prefix_sum.hlsl
//  MX2LM D3D12 Fused Runtime
//
//  Two-entry parallel exclusive prefix sum (Blelloch scan).
//
//  CSScan        — per-block scan, saves block totals
//  CSAddBlockSums — adds block totals back to each element
//
//  Usage (for N elements):
//    Dispatch CSScan        ceil(N/512) groups
//    CPU: prefix-sum the blockSums array (or recurse)
//    Dispatch CSAddBlockSums ceil(N/512) groups
// ============================================================

#include "common.hlsli"

cbuffer RootConstants : register(b0)
{
    uint elementCount;
    uint pad0;
    uint pad1;
    uint pad2;
}

RWStructuredBuffer<uint> data      : register(u0);
RWStructuredBuffer<uint> blockSums : register(u1);

groupshared uint gs[512];

// ── Pass 1: Block-level exclusive scan ───────────────────────
[numthreads(256, 1, 1)]
void CSScan(uint3 gtid : SV_GroupThreadID,
            uint3 gid  : SV_GroupID)
{
    uint base = gid.x * 512u;
    uint a    = base + gtid.x * 2u;
    uint b    = a + 1u;

    gs[gtid.x * 2u]      = (a < elementCount) ? data[a] : 0u;
    gs[gtid.x * 2u + 1u] = (b < elementCount) ? data[b] : 0u;
    GroupMemoryBarrierWithGroupSync();

    // Up-sweep
    uint offset = 1u;
    for (uint d = 256u; d > 0u; d >>= 1)
    {
        GroupMemoryBarrierWithGroupSync();
        if (gtid.x < d)
        {
            uint ai = offset * (gtid.x * 2u + 1u) - 1u;
            uint bi = offset * (gtid.x * 2u + 2u) - 1u;
            gs[bi] += gs[ai];
        }
        offset <<= 1;
    }

    // Save block total, clear last element
    if (gtid.x == 0u) {
        blockSums[gid.x] = gs[511];
        gs[511]          = 0u;
    }

    // Down-sweep
    for (uint d = 1u; d < 512u; d <<= 1)
    {
        offset >>= 1;
        GroupMemoryBarrierWithGroupSync();
        if (gtid.x < d)
        {
            uint ai = offset * (gtid.x * 2u + 1u) - 1u;
            uint bi = offset * (gtid.x * 2u + 2u) - 1u;
            uint tmp = gs[ai];
            gs[ai]   = gs[bi];
            gs[bi]  += tmp;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (a < elementCount) data[a] = gs[gtid.x * 2u];
    if (b < elementCount) data[b] = gs[gtid.x * 2u + 1u];
}

// ── Pass 2: Add block prefix sums back ───────────────────────
// Dispatch: same grid as CSScan (ceil(N/512))
[numthreads(512, 1, 1)]
void CSAddBlockSums(uint3 tid : SV_DispatchThreadID,
                    uint3 gid : SV_GroupID)
{
    uint i = tid.x;
    if (i >= elementCount) return;

    // Block 0's offset is 0 (prefix sum of blockSums[0..gid.x-1])
    // blockSums has been prefix-summed on CPU between the two passes.
    data[i] += blockSums[gid.x];
}
