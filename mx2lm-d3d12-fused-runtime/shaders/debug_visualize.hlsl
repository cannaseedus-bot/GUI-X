// ============================================================
//  debug_visualize.hlsl
//  MX2LM D3D12 Fused Runtime
//
//  Debug overlay: visualize entity forces and signal routing
//  as colored point sprites on a 2D canvas.
// ============================================================

#include "common.hlsli"

cbuffer RootConstants : register(b0)
{
    uint  entityCount;
    uint  canvasWidth;
    uint  canvasHeight;
    float viewScale;
    float viewOffsetX;
    float viewOffsetY;
    float forceVizScale;
    uint  vizMode;      // 0=force, 1=signal, 2=events, 3=velocity
}

StructuredBuffer<float4>  position    : register(t1);
StructuredBuffer<float4>  force       : register(t5);
StructuredBuffer<float>   signalOut   : register(t3);
StructuredBuffer<uint>    events      : register(t9);
StructuredBuffer<float4>  velocity    : register(t2);

RWTexture2D<float4>       canvas      : register(u0);

// ── Color Maps ────────────────────────────────────────────────

float3 HeatMap(float t)
{
    t = saturate(t);
    return float3(
        smoothstep(0.5f, 1.0f, t),
        smoothstep(0.0f, 0.5f, t) * (1.0f - smoothstep(0.5f, 1.0f, t)) * 2.0f,
        smoothstep(0.0f, 0.5f, t)
    );
}

float3 ExpertColor(uint expertId, uint numExperts)
{
    float hue = (float)expertId / (float)max(numExperts, 1u);
    // Simple HSV→RGB (S=1, V=1)
    float3 col = saturate(abs(fmod(hue * 6.0f + float3(0,4,2), 6.0f) - 3.0f) - 1.0f);
    return col;
}

// ── Main ──────────────────────────────────────────────────────

[numthreads(128, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= entityCount) return;

    float4 pos = position[i];

    // Project to 2D canvas
    float2 screenPos = (pos.xz * viewScale) + float2(viewOffsetX, viewOffsetY);
    int2   pixel     = (int2)floor(screenPos);

    if (pixel.x < 0 || pixel.x >= (int)canvasWidth ||
        pixel.y < 0 || pixel.y >= (int)canvasHeight) return;

    float3 color = float3(1, 1, 1);
    float  alpha = 1.0f;

    if (vizMode == 0u)
    {
        // Force magnitude heat map
        float mag = force[i].w / (forceVizScale + EPSILON);
        color = HeatMap(mag);
    }
    else if (vizMode == 1u)
    {
        // Signal / expert routing
        uint expert = (uint)signalOut[i];
        color = ExpertColor(expert, 8u);
    }
    else if (vizMode == 2u)
    {
        // Events: bright white flash on event, else dim
        color = (events[i] > 0u)
              ? float3(1.0f, 0.8f, 0.2f)
              : float3(0.2f, 0.2f, 0.2f);
    }
    else
    {
        // Velocity magnitude
        float speed = length(velocity[i].xyz) * forceVizScale;
        color = HeatMap(speed);
    }

    // Write pixel (additive blend for density visualization)
    canvas[pixel] += float4(color * alpha, alpha);
}
