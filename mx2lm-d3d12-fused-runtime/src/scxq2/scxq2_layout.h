#pragma once
#include <cstdint>

namespace mx2lm {

// SCXQ2 binary format — matches spec/scxq2_layout.md exactly

#pragma pack(push, 1)

struct SCXQ2Header {
    uint8_t  magic[4];           // "SXQ2"
    uint16_t version_major;      // 0
    uint16_t version_minor;      // 1
    uint32_t entity_count;
    uint32_t lane_count;
    uint32_t dict_offset;
    uint32_t dict_size;
    uint32_t fieldmap_offset;
    uint32_t lanes_offset;
    uint32_t payload_offset;
    uint64_t payload_size;
    uint8_t  reserved[6];        // must be zero
};
static_assert(sizeof(SCXQ2Header) == 48, "SCXQ2Header size mismatch");

struct FieldDescriptor {
    char     name[32];
    uint8_t  gpu_register;
    uint8_t  element_size;
    uint8_t  components;
    uint8_t  dtype;              // 0=float32, 1=uint32, 2=int32
    uint32_t lane_index;
    uint8_t  reserved[12];
};
static_assert(sizeof(FieldDescriptor) == 52, "FieldDescriptor size mismatch");

struct LaneDescriptor {
    uint64_t offset;
    uint64_t size;
    uint32_t stride;
    uint32_t count;
    uint8_t  compressed;         // 0=raw, 1=lz4, 2=zstd
    uint8_t  reserved[7];
};
static_assert(sizeof(LaneDescriptor) == 32, "LaneDescriptor size mismatch");

#pragma pack(pop)

// dtype enum
enum class SCXQ2Dtype : uint8_t {
    Float32 = 0,
    UInt32  = 1,
    Int32   = 2,
};

// Standard lane names
constexpr const char* LANE_ENTITIES  = "entities";
constexpr const char* LANE_POSITION  = "position";
constexpr const char* LANE_VELOCITY  = "velocity";
constexpr const char* LANE_SIGNAL    = "signal";
constexpr const char* LANE_AXES_R0   = "axes_row0";
constexpr const char* LANE_AXES_R1   = "axes_row1";
constexpr const char* LANE_AXES_R2   = "axes_row2";

} // namespace mx2lm
