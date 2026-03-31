// ============================================================
//  grid_build.hlsl
//  MX2LM D3D12 Fused Runtime
//
//  Three-pass spatial grid construction:
//    Pass 0  — count entities per cell
//    Pass 1  — exclusive prefix sum (scan)
//    Pass 2  — scatter entity indices
//
//  Each pass is a separate Dispatch with a barrier in between.
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
    uint  passIndex;    // 0 = count, 1 = prefix sum, 2 = scatter
    uint  gridCellCount;
    uint  pad0;
    uint  pad1;
}

StructuredBuffer<float4>  position      : register(t1);
RWStructuredBuffer<uint>  gridCounts    : register(u0);
RWStructuredBuffer<uint>  gridOffsets   : register(u1);
RWStructuredBuffer<uint>  gridIndices   : register(u2);
RWStructuredBuffer<uint>  scatterCursor : register(u3);  // temp per-cell write cursor

// ── Helpers ───────────────────────────────────────────────────

uint GetCellId(float3 pos)
{
    int3 cell = (int3)floor((pos - float3(gridOriginX, gridOriginY, gridOriginZ)) / cellSize);
    cell = clamp(cell, int3(0,0,0), int3((int)gridDimX-1, (int)gridDimY-1, (int)gridDimZ-1));
    return (uint)cell.x
         + (uint)cell.y * gridDimX
         + (uint)cell.z * gridDimX * gridDimY;
}

// ── Pass 0: Count ─────────────────────────────────────────────
// Dispatch: ceil(entityCount / 128)
[numthreads(128, 1, 1)]
void CSCount(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= entityCount) return;

    uint cellId = GetCellId(position[i].xyz);
    InterlockedAdd(gridCounts[cellId], 1u);
}

// ── Pass 1: Exclusive Prefix Sum ─────────────────────────────
// Simple single-threaded scan for small grids.
// For large grids (>64K cells), replace with parallel scan.
// Dispatch: (1, 1, 1)
[numthreads(1, 1, 1)]
void CSScan(uint3 tid : SV_DispatchThreadID)
{
    uint running = 0u;
    for (uint c = 0u; c < gridCellCount; ++c)
    {
        gridOffsets[c]   = running;
        scatterCursor[c] = running;          // initialize scatter cursor
        running         += gridCounts[c];
    }
}

// ── Pass 2: Scatter ───────────────────────────────────────────
// Dispatch: ceil(entityCount / 128)
[numthreads(128, 1, 1)]
void CSScatter(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= entityCount) return;

    uint cellId = GetCellId(position[i].xyz);

    uint slot;
    InterlockedAdd(scatterCursor[cellId], 1u, slot);
    gridIndices[slot] = i;
}
