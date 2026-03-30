#pragma once
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cstdint>

using Microsoft::WRL::ComPtr;

namespace mx2lm {

// Throws std::runtime_error on HRESULT failure
void ThrowIfFailed(HRESULT hr, const char* msg = nullptr);

class DX12Context {
public:
    DX12Context() = default;
    ~DX12Context();

    void Init(bool enableDebugLayer = false);
    void Flush();
    void WaitIdle();

    // Device + queue
    ID3D12Device*              Device()       const { return m_device.Get(); }
    ID3D12CommandQueue*        ComputeQueue() const { return m_computeQueue.Get(); }

    // Per-frame command list management
    void ResetCommandList();
    void ExecuteAndWait();

    ID3D12GraphicsCommandList* CmdList()      const { return m_cmdList.Get(); }

    // Fence utilities
    void SignalFence();
    void WaitForFence();

    uint64_t FenceValue() const { return m_fenceValue; }

private:
    ComPtr<ID3D12Device>              m_device;
    ComPtr<IDXGIFactory6>             m_factory;
    ComPtr<IDXGIAdapter1>             m_adapter;

    ComPtr<ID3D12CommandQueue>        m_computeQueue;
    ComPtr<ID3D12CommandAllocator>    m_cmdAllocator;
    ComPtr<ID3D12GraphicsCommandList> m_cmdList;

    ComPtr<ID3D12Fence>               m_fence;
    HANDLE                            m_fenceEvent = nullptr;
    uint64_t                          m_fenceValue  = 0;
};

} // namespace mx2lm
