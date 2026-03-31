// ============================================================
//  main.cpp  —  MX2LM D3D12 Fused Runtime
//  Real bootstrap: DX12 init + DXC inline compilation + dispatch
// ============================================================

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <dxcapi.h>
#include <wrl/client.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>

using Microsoft::WRL::ComPtr;

// ── Error check ───────────────────────────────────────────────
static void Check(HRESULT hr, const char* msg) {
    if (FAILED(hr)) {
        std::string s = std::string("[FAIL] ") + msg
                      + "  HRESULT=0x" + std::to_string((unsigned)hr);
        throw std::runtime_error(s);
    }
}
#define CHECK(hr, msg) Check((hr), (msg))

// ── GPU buffer helper ─────────────────────────────────────────
struct GPUBuffer {
    ComPtr<ID3D12Resource> resource;
    UINT64                 byteSize = 0;

    void Alloc(ID3D12Device* dev, UINT64 size,
               D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
    {
        byteSize = (size + 15ull) & ~15ull;

        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;

        D3D12_RESOURCE_DESC rd{};
        rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width            = byteSize;
        rd.Height           = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels        = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags            = flags;

        CHECK(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
              D3D12_RESOURCE_STATE_COMMON, nullptr,
              IID_PPV_ARGS(&resource)), "GPUBuffer::Alloc");
    }
};

// ── Globals ───────────────────────────────────────────────────
static ComPtr<ID3D12Device>              g_device;
static ComPtr<ID3D12CommandQueue>        g_queue;
static ComPtr<ID3D12CommandAllocator>    g_allocator;
static ComPtr<ID3D12GraphicsCommandList> g_cmdList;
static ComPtr<ID3D12Fence>              g_fence;
static HANDLE                           g_fenceEvent = nullptr;
static UINT64                           g_fenceValue = 0;

// ── DX12 init ─────────────────────────────────────────────────
static void InitDevice(bool debugLayer) {
#ifdef _DEBUG
    if (debugLayer) {
        ComPtr<ID3D12Debug> dbg;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg))))
            dbg->EnableDebugLayer();
    }
#endif

    ComPtr<IDXGIFactory6> factory;
    CHECK(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");

    // Pick best adapter
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0;
         factory->EnumAdapterByGpuPreference(
             i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
         ++i)
    {
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(),
                                        D3D_FEATURE_LEVEL_12_0,
                                        IID_PPV_ARGS(&g_device))))
            break;
        adapter.Reset();
    }
    if (!g_device)
        throw std::runtime_error("No D3D12-capable adapter (FL 12.0+) found");

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    CHECK(g_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&g_queue)),
          "CreateCommandQueue");

    CHECK(g_device->CreateCommandAllocator(
          D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&g_allocator)),
          "CreateCommandAllocator");

    CHECK(g_device->CreateCommandList(
          0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
          g_allocator.Get(), nullptr, IID_PPV_ARGS(&g_cmdList)),
          "CreateCommandList");

    CHECK(g_cmdList->Close(), "CmdList initial close");

    CHECK(g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence)),
          "CreateFence");

    g_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!g_fenceEvent) throw std::runtime_error("CreateEvent failed");
}

// ── GPU sync ──────────────────────────────────────────────────
static void WaitGPU() {
    ++g_fenceValue;
    CHECK(g_queue->Signal(g_fence.Get(), g_fenceValue), "Queue::Signal");
    if (g_fence->GetCompletedValue() < g_fenceValue) {
        g_fence->SetEventOnCompletion(g_fenceValue, g_fenceEvent);
        WaitForSingleObject(g_fenceEvent, INFINITE);
    }
}

// ── DXC shader compilation ────────────────────────────────────
// Returns compiled bytecode blob (IDxcBlob).
// Caller uses ->GetBufferPointer() / ->GetBufferSize().
static ComPtr<IDxcBlob> CompileShader(const wchar_t* path,
                                       const wchar_t* entry,
                                       const wchar_t* target = L"cs_6_0")
{
    ComPtr<IDxcUtils>    utils;
    ComPtr<IDxcCompiler3> compiler;
    CHECK(DxcCreateInstance(CLSID_DxcUtils,    IID_PPV_ARGS(&utils)),    "DxcUtils");
    CHECK(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler)), "DxcCompiler");

    // Read source
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::wstring ws(path);
        throw std::runtime_error("Cannot open shader: " +
                                 std::string(ws.begin(), ws.end()));
    }
    std::string code((std::istreambuf_iterator<char>(file)), {});

    DxcBuffer buf{};
    buf.Ptr      = code.data();
    buf.Size     = code.size();
    buf.Encoding = DXC_CP_UTF8;

    LPCWSTR args[] = {
        L"-T", target,
        L"-E", entry,
        L"-O3",
        L"-I", L"shaders"   // include path for common.hlsli
    };

    ComPtr<IDxcResult> result;
    CHECK(compiler->Compile(&buf, args, _countof(args), nullptr,
                            IID_PPV_ARGS(&result)), "DxcCompiler::Compile");

    // Check for errors
    ComPtr<IDxcBlobUtf8> errors;
    result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
    if (errors && errors->GetStringLength() > 0)
        std::cerr << "[DXC] " << errors->GetStringPointer() << "\n";

    HRESULT hr = S_OK;
    result->GetStatus(&hr);
    if (FAILED(hr)) throw std::runtime_error("Shader compilation failed");

    ComPtr<IDxcBlob> blob;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&blob), nullptr);
    return blob;
}

// ── UAV barrier ───────────────────────────────────────────────
static void UAVBarrier(ID3D12Resource* res) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = res;
    g_cmdList->ResourceBarrier(1, &b);
}

// ── Main ──────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    bool debugLayer    = false;
    bool useScxq2      = false;
    const char* scxq2  = nullptr;
    UINT entityCount   = 1024;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--debug"))         debugLayer  = true;
        else if (!strcmp(argv[i], "--scxq2") && i + 1 < argc) {
            scxq2    = argv[++i];
            useScxq2 = true;
        }
        else if (!strcmp(argv[i], "--entities") && i + 1 < argc)
            entityCount = (UINT)atoi(argv[++i]);
    }

    std::cout << "==============================================\n"
              << "  MX2LM D3D12 Fused Runtime  (bootstrap)\n"
              << "  entities=" << entityCount << "\n"
              << "==============================================\n\n";

    try {
        // ── Device ────────────────────────────────────────────
        InitDevice(debugLayer);
        std::cout << "[OK] Device\n";

        // ── Compile shaders ───────────────────────────────────
        auto gridCountShader  = CompileShader(L"shaders/grid_build.hlsl",  L"CSCount",   L"cs_6_0");
        auto gridScanShader   = CompileShader(L"shaders/grid_build.hlsl",  L"CSScan",    L"cs_6_0");
        auto gridScatterShader= CompileShader(L"shaders/grid_build.hlsl",  L"CSScatter", L"cs_6_0");
        auto fusedShader      = CompileShader(L"shaders/fused_attention_force_moe.hlsl", L"CSMain", L"cs_6_0");
        std::cout << "[OK] Shaders compiled\n";

        // ── Root signature (minimal — inline constants via b0) ─
        D3D12_ROOT_SIGNATURE_DESC rsDesc{};
        rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

        ComPtr<ID3DBlob> sigBlob, errBlob;
        CHECK(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1,
                                         &sigBlob, &errBlob),
              "SerializeRootSignature");

        ComPtr<ID3D12RootSignature> rootSig;
        CHECK(g_device->CreateRootSignature(0,
              sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(),
              IID_PPV_ARGS(&rootSig)), "CreateRootSignature");

        // ── PSO factory ───────────────────────────────────────
        auto MakePSO = [&](ComPtr<IDxcBlob>& blob) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC d{};
            d.pRootSignature     = rootSig.Get();
            d.CS.pShaderBytecode = blob->GetBufferPointer();
            d.CS.BytecodeLength  = blob->GetBufferSize();
            ComPtr<ID3D12PipelineState> pso;
            CHECK(g_device->CreateComputePipelineState(&d, IID_PPV_ARGS(&pso)),
                  "CreateComputePipelineState");
            return pso;
        };

        auto psoCount   = MakePSO(gridCountShader);
        auto psoScan    = MakePSO(gridScanShader);
        auto psoScatter = MakePSO(gridScatterShader);
        auto psoFused   = MakePSO(fusedShader);
        std::cout << "[OK] PSOs created\n";

        // ── Allocate minimal GPU buffers ──────────────────────
        // (Full BufferManager path used when building with full class stack)
        UINT gridCells = 128 * 128 * 128;
        GPUBuffer positionBuf, gridCountBuf, gridIndexBuf, forceBuf;
        positionBuf  .Alloc(g_device.Get(), entityCount * 16,  D3D12_RESOURCE_FLAG_NONE);
        gridCountBuf .Alloc(g_device.Get(), gridCells   * 4);
        gridIndexBuf .Alloc(g_device.Get(), entityCount * 4);
        forceBuf     .Alloc(g_device.Get(), entityCount * 16);
        std::cout << "[OK] Buffers allocated\n";

        // ── Record + dispatch ─────────────────────────────────
        CHECK(g_allocator->Reset(), "CmdAllocator::Reset");
        CHECK(g_cmdList->Reset(g_allocator.Get(), nullptr), "CmdList::Reset");

        UINT groups = (entityCount + 127u) / 128u;

        // Pass 0 — grid count
        g_cmdList->SetComputeRootSignature(rootSig.Get());
        g_cmdList->SetPipelineState(psoCount.Get());
        g_cmdList->Dispatch(groups, 1, 1);
        UAVBarrier(gridCountBuf.resource.Get());

        // Pass 1 — prefix scan (1 group for small grids)
        g_cmdList->SetPipelineState(psoScan.Get());
        g_cmdList->Dispatch(1, 1, 1);
        UAVBarrier(gridCountBuf.resource.Get());

        // Pass 2 — scatter
        g_cmdList->SetPipelineState(psoScatter.Get());
        g_cmdList->Dispatch(groups, 1, 1);
        UAVBarrier(gridIndexBuf.resource.Get());

        // Fused kernel
        g_cmdList->SetPipelineState(psoFused.Get());
        g_cmdList->Dispatch(groups, 1, 1);
        UAVBarrier(forceBuf.resource.Get());

        CHECK(g_cmdList->Close(), "CmdList::Close");

        ID3D12CommandList* lists[] = { g_cmdList.Get() };
        g_queue->ExecuteCommandLists(1, lists);
        WaitGPU();

        std::cout << "[OK] Dispatch complete\n\n"
                  << "RUN COMPLETE\n";
    }
    catch (const std::exception& e) {
        std::cerr << "\n[FATAL] " << e.what() << "\n";
        if (g_fenceEvent) CloseHandle(g_fenceEvent);
        return 1;
    }

    if (g_fenceEvent) CloseHandle(g_fenceEvent);
    return 0;
}
