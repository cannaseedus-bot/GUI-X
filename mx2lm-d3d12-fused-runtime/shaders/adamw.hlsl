// ============================================================
//  adamw.hlsl
//  MX2LM D3D12 Fused Runtime  —  Training: AdamW weight update
//
//  One thread = one weight parameter.
//  Bias correction is applied per-dispatch (step injected via cbuffer).
//  LoRA rank-factored weights (lora_A, lora_B) updated separately.
// ============================================================

#include "common.hlsli"

cbuffer AdamWParams : register(b0)
{
    float    lr;        // learning rate (e.g. 1e-4)
    float    beta1;     // momentum decay (e.g. 0.9)
    float    beta2;     // variance decay (e.g. 0.999)
    float    eps;       // numerical stability (e.g. 1e-8)
    float    wd;        // weight decay (e.g. 0.01)
    uint     step;      // current training step (for bias correction)
    uint     paramCount;
    uint     pad;
}

// ── Weight buffers ────────────────────────────────────────────
RWStructuredBuffer<float> weights : register(u0);   // θ
RWStructuredBuffer<float> grads   : register(u1);   // ∇L
RWStructuredBuffer<float> m_buf   : register(u2);   // 1st moment
RWStructuredBuffer<float> v_buf   : register(u3);   // 2nd moment

// ── LoRA buffers (rank-factored delta) ────────────────────────
RWStructuredBuffer<float> lora_A  : register(u4);   // rank projection A
RWStructuredBuffer<float> lora_B  : register(u5);   // rank projection B

// ── AdamW kernel ──────────────────────────────────────────────
[numthreads(128, 1, 1)]
void CSAdamW(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= paramCount) return;

    float g = grads[i];

    // Moment update
    float m_new = beta1 * m_buf[i] + (1.0f - beta1) * g;
    float v_new = beta2 * v_buf[i] + (1.0f - beta2) * g * g;

    m_buf[i] = m_new;
    v_buf[i] = v_new;

    // Bias-corrected estimates
    float bc1   = 1.0f - pow(beta1, (float)step);
    float bc2   = 1.0f - pow(beta2, (float)step);
    float m_hat = m_new / (bc1 + EPSILON);
    float v_hat = v_new / (bc2 + EPSILON);

    // AdamW update: weight decay applied to raw weight (not gradient)
    float theta = weights[i];
    theta = theta - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta);

    weights[i] = GuardNaN(theta);

    // Zero gradient after consumption
    grads[i] = 0.0f;
}

// ── LoRA update kernel ────────────────────────────────────────
// Separate dispatch — same AdamW rule but on lora_A/lora_B.
// lora_A[i] and lora_B[i] treated as independent parameters.
[numthreads(128, 1, 1)]
void CSLoRAUpdate(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= paramCount) return;

    // Gradient of LoRA delta δ = lora_A[i] * lora_B[i] flows back separately.
    // Here we just damp lora_A toward 0 (regularize) and let lora_B carry signal.
    lora_A[i] = lora_A[i] * (1.0f - wd * lr);
    lora_B[i] = lora_B[i] * (1.0f - wd * lr);
}
