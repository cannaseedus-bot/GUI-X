#include "scxq2_loader.h"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <random>
#include <DirectXMath.h>

using namespace DirectX;

namespace mx2lm {

// ── File Load ─────────────────────────────────────────────────

void SCXQ2Loader::Load(const std::string& path,
                        DX12Context*       ctx,
                        BufferManager*     bufs)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("SCXQ2: cannot open " + path);

    size_t fileSize = (size_t)file.tellg();
    file.seekg(0);
    std::vector<uint8_t> raw(fileSize);
    file.read((char*)raw.data(), (std::streamsize)fileSize);

    SCXQ2Header hdr{};
    ParseHeader(raw.data(), fileSize, hdr);

    m_entityCount = hdr.entity_count;

    ParseDict(raw.data(), hdr.dict_offset, hdr.dict_size);
    ParseLanes(raw.data(), hdr, hdr.lane_count);

    // Upload all lanes
    ctx->ResetCommandList();
    for (const auto& lane : m_lanes)
        UploadLane(lane, ctx, bufs, ctx->CmdList());
    ctx->ExecuteAndWait();
}

void SCXQ2Loader::ParseHeader(const uint8_t* data,
                               size_t         fileSize,
                               SCXQ2Header&   hdr)
{
    if (fileSize < sizeof(SCXQ2Header))
        throw std::runtime_error("SCXQ2: file too small for header");

    memcpy(&hdr, data, sizeof(SCXQ2Header));

    if (hdr.magic[0] != 'S' || hdr.magic[1] != 'X' ||
        hdr.magic[2] != 'Q' || hdr.magic[3] != '2')
        throw std::runtime_error("SCXQ2: invalid magic — expected 'SXQ2'");

    if (hdr.version_major != 0)
        throw std::runtime_error("SCXQ2: unsupported version "
                                 + std::to_string(hdr.version_major));

    if (hdr.payload_offset + hdr.payload_size > (uint64_t)fileSize)
        throw std::runtime_error("SCXQ2: payload extends beyond file end");
}

void SCXQ2Loader::ParseDict(const uint8_t* data,
                              uint32_t       offset,
                              uint32_t       size)
{
    const char* ptr = (const char*)(data + offset);
    const char* end = ptr + size;

    while (ptr < end && *ptr) {
        std::string kv(ptr);
        ptr += kv.size() + 1;

        size_t eq = kv.find('=');
        if (eq != std::string::npos)
            m_dict[kv.substr(0, eq)] = kv.substr(eq + 1);
    }
}

void SCXQ2Loader::ParseLanes(const uint8_t*     data,
                               const SCXQ2Header& hdr,
                               uint32_t           laneCount)
{
    const FieldDescriptor* fields =
        (const FieldDescriptor*)(data + hdr.fieldmap_offset);
    const LaneDescriptor* lanes =
        (const LaneDescriptor*)(data + hdr.lanes_offset);

    for (uint32_t i = 0; i < laneCount; ++i) {
        ParsedLane pl;
        pl.name  = std::string(fields[i].name,
                               strnlen(fields[i].name, sizeof(fields[i].name)));
        pl.field = fields[i];
        pl.desc  = lanes[i];

        if (pl.desc.compressed != 0)
            throw std::runtime_error("SCXQ2: lane '" + pl.name
                                     + "': compressed lanes not yet supported");

        uint64_t off  = hdr.payload_offset + pl.desc.offset;
        uint64_t sz   = pl.desc.size;

        if (off + sz > hdr.payload_offset + hdr.payload_size)
            throw std::runtime_error("SCXQ2: lane '" + pl.name
                                     + "' extends beyond payload");

        pl.data.assign(data + off, data + off + sz);
        m_lanes.push_back(std::move(pl));
    }
}

void SCXQ2Loader::UploadLane(const ParsedLane&          lane,
                               DX12Context*               ctx,
                               BufferManager*             bufs,
                               ID3D12GraphicsCommandList* cmdList)
{
    auto& b = bufs->Buffers();
    ID3D12Resource* dst = nullptr;

    if      (lane.name == LANE_ENTITIES) dst = b.entities.Resource();
    else if (lane.name == LANE_POSITION) dst = b.position.Resource();
    else if (lane.name == LANE_VELOCITY) dst = b.velocity.Resource();
    else if (lane.name == LANE_SIGNAL)   dst = b.signal.Resource();
    else if (lane.name == LANE_AXES_R0 ||
             lane.name == LANE_AXES_R1 ||
             lane.name == LANE_AXES_R2) dst = b.axes.Resource();
    else return;  // unknown lane — skip

    // Axes rows are packed consecutively in the axes buffer:
    //   row0 at offset 0, row1 at offset N*16, row2 at offset N*32
    uint64_t byteOffset = 0;
    if (lane.name == LANE_AXES_R1) byteOffset = (uint64_t)m_entityCount * sizeof(XMFLOAT4);
    if (lane.name == LANE_AXES_R2) byteOffset = (uint64_t)m_entityCount * sizeof(XMFLOAT4) * 2;

    bufs->Upload(cmdList, dst, lane.data.data(), (uint64_t)lane.data.size());
    (void)byteOffset;  // used when splitting axes rows into sub-regions
}

// ── Random initial state generator ───────────────────────────

void SCXQ2Loader::GenerateRandom(uint32_t       entityCount,
                                   DX12Context*   ctx,
                                   BufferManager* bufs)
{
    m_entityCount = entityCount;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos(-30.0f, 30.0f);
    std::uniform_real_distribution<float> vel(-1.0f,  1.0f);
    std::uniform_real_distribution<float> sig(0.0f,   8.0f);

    std::vector<uint32_t> entities(entityCount);
    std::vector<XMFLOAT4> positions(entityCount);
    std::vector<XMFLOAT4> velocities(entityCount);
    std::vector<float>    signals(entityCount);
    std::vector<XMFLOAT4> axes(entityCount * 3u);

    for (uint32_t i = 0; i < entityCount; ++i) {
        entities[i]   = i;
        positions[i]  = { pos(rng), pos(rng), pos(rng), 1.0f };
        velocities[i] = { vel(rng), vel(rng), vel(rng), 1.0f };
        signals[i]    = sig(rng);

        // Identity orientation
        axes[i * 3u + 0u] = { 1.f, 0.f, 0.f, 0.f };
        axes[i * 3u + 1u] = { 0.f, 1.f, 0.f, 0.f };
        axes[i * 3u + 2u] = { 0.f, 0.f, 1.f, 0.f };
    }

    auto& b = bufs->Buffers();
    ctx->ResetCommandList();
    auto* cmd = ctx->CmdList();

    bufs->Upload(cmd, b.entities.Resource(),
                 entities.data(),   entities.size()   * sizeof(uint32_t));
    bufs->Upload(cmd, b.position.Resource(),
                 positions.data(),  positions.size()  * sizeof(XMFLOAT4));
    bufs->Upload(cmd, b.velocity.Resource(),
                 velocities.data(), velocities.size() * sizeof(XMFLOAT4));
    bufs->Upload(cmd, b.signal.Resource(),
                 signals.data(),    signals.size()    * sizeof(float));
    bufs->Upload(cmd, b.axes.Resource(),
                 axes.data(),       axes.size()       * sizeof(XMFLOAT4));

    ctx->ExecuteAndWait();
}

} // namespace mx2lm
