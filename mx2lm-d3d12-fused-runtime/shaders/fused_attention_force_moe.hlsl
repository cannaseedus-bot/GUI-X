// ============================================================
//  fused_attention_force_moe.hlsl
//  MX2LM D3D12 Fused Runtime
//
//  Single-pass kernel: QKV → attention → force → MoE routing
//  → event emission
//
//  SM 6.0  |  Shader Model 6.0 minimum
//  [numthreads(128, 1, 1)]
// ============================================================

#include "common.hlsli"

// ── Root Constants ────────────────────────────────────────────
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
    float forceMax;
    uint  neighborCap;
    uint  numExperts;
    float eventThreshold;
}

// ── SRV Inputs ────────────────────────────────────────────────
StructuredBuffer<uint>    entities      : register(t0);
StructuredBuffer<float4>  position      : register(t1);
StructuredBuffer<float4>  velocity      : register(t2);
StructuredBuffer<float>   signalIn      : register(t3);
StructuredBuffer<float4>  axesData      : register(t4);  // 3 float4 per entity
StructuredBuffer<uint>    gridOffsets   : register(t6);
StructuredBuffer<uint>    gridCounts    : register(t7);
StructuredBuffer<uint>    gridIndices   : register(t8);

// ── UAV Outputs ───────────────────────────────────────────────
RWStructuredBuffer<float4> force        : register(u5);
RWStructuredBuffer<float>  signalOut    : register(u3);
RWStructuredBuffer<uint>   events       : register(u9);
RWStructuredBuffer<float4> eventParams  : register(u10);

// ── Learned Weight Constants (small inline matrices) ─────────
// For MVP: 4-dim projection (extend to larger dims later)
static const float Wq[4][4] = {
    { 1.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 1.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 1.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f }
};
static const float Wk[4][4] = {
    { 1.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 1.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 1.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 1.0f }
};
static const float Wv[4][4] = {
    { 0.5f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.5f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.5f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.5f }
};
static const float Wforce[4][4] = {
    { 1.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 1.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 1.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f }
};

// ── Helpers ───────────────────────────────────────────────────

float4 MatMul4x4(float m[4][4], float4 v)
{
    return float4(
        dot(float4(m[0][0], m[0][1], m[0][2], m[0][3]), v),
        dot(float4(m[1][0], m[1][1], m[1][2], m[1][3]), v),
        dot(float4(m[2][0], m[2][1], m[2][2], m[2][3]), v),
        dot(float4(m[3][0], m[3][1], m[3][2], m[3][3]), v)
    );
}

float3 SafeNormalize3(float3 v)
{
    float len = length(v);
    return (len > EPSILON) ? (v / len) : float3(0.0f, 0.0f, 1.0f);
}

float GuardNaN(float v)
{
    return (isnan(v) || isinf(v)) ? 0.0f : v;
}

float4 GuardNaN4(float4 v)
{
    return float4(GuardNaN(v.x), GuardNaN(v.y), GuardNaN(v.z), GuardNaN(v.w));
}

uint GetCellId(float3 pos)
{
    int3 cell = (int3)floor((pos - float3(gridOriginX, gridOriginY, gridOriginZ)) / cellSize);
    cell = clamp(cell, int3(0,0,0), int3((int)gridDimX-1, (int)gridDimY-1, (int)gridDimZ-1));
    return (uint)cell.x
         + (uint)cell.y * gridDimX
         + (uint)cell.z * gridDimX * gridDimY;
}

// Retrieve axes rows for entity (3 float4 stored consecutively)
void GetAxes(uint idx, out float3 ax, out float3 ay, out float3 az)
{
    uint base = idx * 3u;
    ax = axesData[base + 0].xyz;
    ay = axesData[base + 1].xyz;
    az = axesData[base + 2].xyz;
}

// ── Main Kernel ───────────────────────────────────────────────

[numthreads(128, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= entityCount) return;

    // ── Load entity state ─────────────────────────────────────
    float4 pos_i = position[i];
    float4 vel_i = velocity[i];
    float  sig_i = signalIn[i];

    float3 ax, ay, az;
    GetAxes(i, ax, ay, az);

    // ── Q projection ──────────────────────────────────────────
    float4 q = MatMul4x4(Wq, pos_i);
    float  invSqrtDim = 0.5f;  // 1/sqrt(4)

    // ── Neighbor sampling ─────────────────────────────────────
    uint   myCell    = GetCellId(pos_i.xyz);
    int3   myCoord   = (int3)floor((pos_i.xyz - float3(gridOriginX, gridOriginY, gridOriginZ)) / cellSize);

    float4 contextVec = float4(0, 0, 0, 0);
    float  totalWeight = 0.0f;
    uint   neighborCount = 0u;

    // Gather scores + values for softmax
    float scores[64];
    float4 values[64];

    [loop]
    for (int dz = -1; dz <= 1 && neighborCount < neighborCap; ++dz)
    [loop]
    for (int dy = -1; dy <= 1 && neighborCount < neighborCap; ++dy)
    [loop]
    for (int dx = -1; dx <= 1 && neighborCount < neighborCap; ++dx)
    {
        int3 nc = myCoord + int3(dx, dy, dz);
        if (any(nc < 0) || nc.x >= (int)gridDimX ||
            nc.y >= (int)gridDimY || nc.z >= (int)gridDimZ) continue;

        uint cellId = (uint)nc.x
                    + (uint)nc.y * gridDimX
                    + (uint)nc.z * gridDimX * gridDimY;

        uint cellStart = gridOffsets[cellId];
        uint cellCount = gridCounts[cellId];

        [loop]
        for (uint c = 0u; c < cellCount && neighborCount < neighborCap; ++c)
        {
            uint j = gridIndices[cellStart + c];
            if (j == i) continue;

            float4 pos_j = position[j];
            float  sig_j = signalIn[j];

            // K projection
            float4 k = MatMul4x4(Wk, pos_j);
            // V projection (uses signal as additional component)
            float4 v_in = float4(pos_j.xyz, sig_j);
            float4 v    = MatMul4x4(Wv, v_in);

            float score = dot(q, k) * invSqrtDim;
            scores[neighborCount] = score;
            values[neighborCount] = v;
            neighborCount++;
        }
    }

    // ── Softmax over scores ───────────────────────────────────
    if (neighborCount > 0u)
    {
        float maxScore = scores[0];
        for (uint s = 1u; s < neighborCount; ++s)
            maxScore = max(maxScore, scores[s]);

        float sumExp = 0.0f;
        for (uint s = 0u; s < neighborCount; ++s)
        {
            scores[s] = exp(scores[s] - maxScore);
            sumExp   += scores[s];
        }
        sumExp += EPSILON;

        // Weighted sum → context
        for (uint s = 0u; s < neighborCount; ++s)
        {
            float w     = scores[s] / sumExp;
            contextVec += values[s] * w;
        }
    }

    // ── Force accumulation ────────────────────────────────────
    float4 f = MatMul4x4(Wforce, contextVec);

    // Apply local frame rotation
    float3 forceLocal = f.xyz;
    float3 forceWorld = ax * forceLocal.x
                      + ay * forceLocal.y
                      + az * forceLocal.z;

    // Clamp + NaN guard
    forceWorld = clamp(forceWorld, -forceMax, forceMax);
    float forceMag = length(forceWorld);
    float4 forceOut = GuardNaN4(float4(forceWorld, forceMag));

    // ── MoE Routing ───────────────────────────────────────────
    // Gate: dot context with each expert bias (simplified)
    float  bestScore   = -1e9f;
    uint   bestExpert  = 0u;

    for (uint e = 0u; e < numExperts; ++e)
    {
        // Expert e uses component e % 4 of context as gate score
        float gateScore = contextVec[(int)(e % 4u)];
        if (gateScore > bestScore)
        {
            bestScore  = gateScore;
            bestExpert = e;
        }
    }
    float newSignal = (float)bestExpert;

    // ── Event Emission ────────────────────────────────────────
    uint  emittedEvent  = 0u;
    float4 emittedParam = float4(0, 0, 0, 0);

    if (forceMag > eventThreshold)
    {
        emittedEvent  = 1u;  // EVENT_FORCE_SPIKE
        emittedParam  = float4(forceWorld, forceMag);
    }

    // ── Write Outputs ─────────────────────────────────────────
    force[i]       = forceOut;
    signalOut[i]   = GuardNaN(newSignal);
    events[i]      = emittedEvent;
    eventParams[i] = emittedParam;
}
