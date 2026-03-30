#pragma once
#include "scxq2_layout.h"
#include "../buffers/buffer_manager.h"
#include "../dx12_context.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace mx2lm {

struct ParsedLane {
    std::string        name;
    FieldDescriptor    field;
    LaneDescriptor     desc;
    std::vector<uint8_t> data;  // decompressed payload
};

class SCXQ2Loader {
public:
    // Load from file, decompress lanes, upload to GPU buffers
    void Load(const std::string& path,
              DX12Context*       ctx,
              BufferManager*     bufs);

    // Generate a random initial state (no file required)
    void GenerateRandom(uint32_t    entityCount,
                        DX12Context* ctx,
                        BufferManager* bufs);

    const std::unordered_map<std::string, std::string>& Dict() const { return m_dict; }
    uint32_t EntityCount() const { return m_entityCount; }

private:
    std::unordered_map<std::string, std::string> m_dict;
    std::vector<ParsedLane>                       m_lanes;
    uint32_t                                      m_entityCount = 0;

    void ParseHeader(const uint8_t* data, size_t fileSize, SCXQ2Header& hdr);
    void ParseDict(const uint8_t* data, uint32_t offset, uint32_t size);
    void ParseLanes(const uint8_t* data,
                    const SCXQ2Header& hdr,
                    uint32_t laneCount);

    void UploadLane(const ParsedLane& lane,
                    DX12Context*      ctx,
                    BufferManager*    bufs,
                    ID3D12GraphicsCommandList* cmdList);
};

} // namespace mx2lm
