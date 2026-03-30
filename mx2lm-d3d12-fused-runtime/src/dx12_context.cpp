#include "dx12_context.h"
#include <stdexcept>
#include <string>

#ifdef _DEBUG
#include <d3d12sdklayers.h>
#endif

namespace mx2lm {

void ThrowIfFailed(HRESULT hr, const char* msg) {
    if (FAILED(hr)) {
        std::string s = msg ? msg : "D3D12 call failed";
        s += " (HRESULT=0x" + std::to_string((unsigned)hr) + ")";
        throw std::runtime_error(s);
    }
}

// ── Init ──────────────────────────────────────────────────────

void DX12Context::Init(bool enableDebugLayer) {
#ifdef _DEBUG
    if (enableDebugLayer) {
        ComPtr<ID3D12Debug> dbg;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg))))
            dbg->EnableDebugLayer();
    }
#endif

    ThrowIfFailed(CreateDXGIFactory2(0, IID_PPV_ARGS(&m_factory)),
                  "CreateDXGIFactory2");

    // Pick highest-performance adapter
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0;
         m_factory->EnumAdapterByGpuPreference(
             i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
         ++i)
    {
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(),
                                        D3D_FEATURE_LEVEL_11_0,
                                        IID_PPV_ARGS(&m_device))))
        {
            m_adapter = adapter;
            break;
        }
    }

    if (!m_device)
        ThrowIfFailed(E_FAIL, "No D3D12-capable adapter found");

    // Compute queue
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type  = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    qd.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    ThrowIfFailed(m_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&m_computeQueue)),
                  "CreateCommandQueue");

    // Allocator + command list
    ThrowIfFailed(m_device->CreateCommandAllocator(
                      D3D12_COMMAND_LIST_TYPE_COMPUTE,
                      IID_PPV_ARGS(&m_cmdAllocator)),
                  "CreateCommandAllocator");

    ThrowIfFailed(m_device->CreateCommandList(
                      0,
                      D3D12_COMMAND_LIST_TYPE_COMPUTE,
                      m_cmdAllocator.Get(),
                      nullptr,
                      IID_PPV_ARGS(&m_cmdList)),
                  "CreateCommandList");

    ThrowIfFailed(m_cmdList->Close(), "CmdList->Close (initial)");

    // Fence
    ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                        IID_PPV_ARGS(&m_fence)),
                  "CreateFence");

    m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!m_fenceEvent)
        throw std::runtime_error("CreateEvent failed");
}

// ── Command List ──────────────────────────────────────────────

void DX12Context::ResetCommandList() {
    ThrowIfFailed(m_cmdAllocator->Reset(), "CmdAllocator->Reset");
    ThrowIfFailed(m_cmdList->Reset(m_cmdAllocator.Get(), nullptr), "CmdList->Reset");
}

void DX12Context::ExecuteAndWait() {
    ThrowIfFailed(m_cmdList->Close(), "CmdList->Close");

    ID3D12CommandList* lists[] = { m_cmdList.Get() };
    m_computeQueue->ExecuteCommandLists(1, lists);

    SignalFence();
    WaitForFence();
}

// ── Fence ─────────────────────────────────────────────────────

void DX12Context::SignalFence() {
    ++m_fenceValue;
    ThrowIfFailed(m_computeQueue->Signal(m_fence.Get(), m_fenceValue),
                  "Queue->Signal");
}

void DX12Context::WaitForFence() {
    if (m_fence->GetCompletedValue() < m_fenceValue) {
        ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent),
                      "Fence->SetEventOnCompletion");
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }
}

void DX12Context::WaitIdle() {
    SignalFence();
    WaitForFence();
}

void DX12Context::Flush() { WaitIdle(); }

// ── Destructor ────────────────────────────────────────────────

DX12Context::~DX12Context() {
    if (m_device) Flush();
    if (m_fenceEvent) CloseHandle(m_fenceEvent);
}

} // namespace mx2lm
