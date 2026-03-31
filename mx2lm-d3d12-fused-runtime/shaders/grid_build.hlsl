// ============================================================
//  grid_build.hlsl
//  MX2LM D3D12 Fused Runtime
//
//  Three entry points (compiled to separate PSOs):
//    CSCount   — atomic count per cell
//    CSScan    — exclusive prefix sum (single-group, up to 65536 cells)
//    CSScatter — scatter entity indices into sorted slots
// ============================================================

#include "common.hlsli"

cbuffer RootConstants : register(b0)
{
    uint  entityCount;
    uint  gridDimX;
    uint  gridDimY;
    uint  gridDimZ;
    float cellSize;
    float gridOriginX;
    float gridOriginY;
    float gridOriginZ;
    uint  gridCellCount;
    uint  pad0;
    uint  pad1;
    uint  pad2;
}

StructuredBuffer<float4>  position      : register(t0);
RWStructuredBuffer<uint>  gridCounts    : register(u0);
RWStructuredBuffer<uint>  gridOffsets   : register(u1);
RWStructuredBuffer<uint>  gridIndices   : register(u2);
RWStructuredBuffer<uint>  scatterCursor : register(u3);

// ── Shared for CSScan ─────────────────────────────────────────
groupshared uint gs_scan[512];

// ── Helper: cell ID from world position ───────────────────────
uint CellId(float3 pos)
{
    int3 c = (int3)floor((pos - float3(gridOriginX, gridOriginY, gridOriginZ)) / cellSize);
    c = clamp(c, int3(0,0,0), int3((int)gridDimX-1, (int)gridDimY-1, (int)gridDimZ-1));
    return (uint)c.x + (uint)c.y * gridDimX + (uint)c.z * gridDimX * gridDimY;
}

// ── Pass 0: Count ─────────────────────────────────────────────
// Dispatch: ceil(entityCount / 128)
[numthreads(128, 1, 1)]
void CSCount(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= entityCount) return;

    uint cell = CellId(position[i].xyz);
    InterlockedAdd(gridCounts[cell], 1u);
}

// ── Pass 1: Exclusive Prefix Sum (Blelloch, up to 512 cells/group) ──
// For grids larger than 512 cells use prefix_sum.hlsl multi-pass.
// Dispatch: ceil(gridCellCount / 512)
[numthreads(256, 1, 1)]
void CSScan(uint3 gtid : SV_GroupThreadID,
            uint3 gid  : SV_GroupID)
{
    uint base = gid.x * 512u;
    uint a    = base + gtid.x * 2u;
    uint b    = a + 1u;

    gs_scan[gtid.x * 2u]      = (a < gridCellCount) ? gridCounts[a] : 0u;
    gs_scan[gtid.x * 2u + 1u] = (b < gridCellCount) ? gridCounts[b] : 0u;
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
            gs_scan[bi] += gs_scan[ai];
        }
        offset <<= 1;
    }

    if (gtid.x == 0u) gs_scan[511] = 0u;

    // Down-sweep
    for (uint d = 1u; d < 512u; d <<= 1)
    {
        offset >>= 1;
        GroupMemoryBarrierWithGroupSync();
        if (gtid.x < d)
        {
            uint ai = offset * (gtid.x * 2u + 1u) - 1u;
            uint bi = offset * (gtid.x * 2u + 2u) - 1u;
            uint tmp    = gs_scan[ai];
            gs_scan[ai] = gs_scan[bi];
            gs_scan[bi] += tmp;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (a < gridCellCount) {
        gridOffsets[a]   = gs_scan[gtid.x * 2u];
        scatterCursor[a] = gs_scan[gtid.x * 2u];      // init scatter write cursor
    }
    if (b < gridCellCount) {
        gridOffsets[b]   = gs_scan[gtid.x * 2u + 1u];
        scatterCursor[b] = gs_scan[gtid.x * 2u + 1u];
    }
}

// ── Pass 2: Scatter ───────────────────────────────────────────
// Dispatch: ceil(entityCount / 128)
[numthreads(128, 1, 1)]
void CSScatter(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= entityCount) return;

    uint cell = CellId(position[i].xyz);

    uint slot;
    InterlockedAdd(scatterCursor[cell], 1u, slot);
    gridIndices[slot] = i;
}
