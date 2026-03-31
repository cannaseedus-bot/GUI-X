// ============================================================
//  lora.hlsli
//  MX2LM D3D12 Fused Runtime
//  LoRA delta injection — included by fused_attention_force_moe.hlsl
//
//  Usage:
//    #define LORA_ENABLED 1
//    #include "lora.hlsli"
//    ...
//    float4 q = MatMul4x4(Wq, pos_i);
//    q = LoRAInject(q, i);
// ============================================================

#ifndef LORA_HLSLI
#define LORA_HLSLI

#ifdef LORA_ENABLED

// Rank-r factored weight delta: Δ = A·B  (r << d)
// Stored as individual float scalars aligned to entity index.
StructuredBuffer<float> lora_A : register(t11);  // A projection weights
StructuredBuffer<float> lora_B : register(t12);  // B projection weights

cbuffer LoRAParams : register(b1)
{
    float loraScale;   // scaling factor α/r (default: 1.0)
    uint  loraRank;    // rank r
    uint  loraDim;     // output dim (must match QKV dim)
    uint  loraOffset;  // base index into lora_A/lora_B arrays
}

// Compute rank-r LoRA delta for entity i and add to signal.
float LoRADelta(uint entityIdx)
{
    uint base = loraOffset + entityIdx * loraRank;
    float delta = 0.0f;
    for (uint r = 0u; r < loraRank; ++r)
        delta += lora_A[base + r] * lora_B[base + r];
    return delta * loraScale;
}

// Inject LoRA delta into a scalar routing signal.
float LoRAInjectSignal(float signal, uint entityIdx)
{
    return signal + LoRADelta(entityIdx);
}

#else

// No-op when LORA_ENABLED is not defined
float LoRADelta(uint entityIdx)       { return 0.0f; }
float LoRAInjectSignal(float s, uint) { return s; }

#endif  // LORA_ENABLED
#endif  // LORA_HLSLI
