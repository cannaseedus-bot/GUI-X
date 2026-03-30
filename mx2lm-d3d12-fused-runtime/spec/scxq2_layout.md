# SCXQ2 Layout — Binary Tensor + Agent Format

## Overview

SCXQ2 is the **binary serialization format** for agent state and tensor data used
by the MX2LM runtime. It maps directly to GPU structured buffers with zero copy
where possible.

---

## File Structure

```
┌─────────────────────────────────────────────────┐
│  [HEADER]         48 bytes, fixed                │
├─────────────────────────────────────────────────┤
│  [DICT]           variable, null-terminated KV   │
├─────────────────────────────────────────────────┤
│  [FIELD MAP]      lane_count × FieldDescriptor   │
├─────────────────────────────────────────────────┤
│  [LANES]          lane_count × LaneDescriptor    │
├─────────────────────────────────────────────────┤
│  [PAYLOAD]        raw binary tensor data         │
└─────────────────────────────────────────────────┘
```

---

## HEADER (48 bytes)

```c
struct SCXQ2Header {
    uint8_t  magic[4];       // "SXQ2"
    uint16_t version_major;  // 0
    uint16_t version_minor;  // 1
    uint32_t entity_count;   // number of entities
    uint32_t lane_count;     // number of data lanes
    uint32_t dict_offset;    // byte offset to DICT section
    uint32_t dict_size;      // byte size of DICT section
    uint32_t fieldmap_offset;// byte offset to FIELD MAP section
    uint32_t lanes_offset;   // byte offset to LANES section
    uint32_t payload_offset; // byte offset to PAYLOAD section
    uint64_t payload_size;   // total payload size in bytes
    uint8_t  reserved[6];    // must be zero
};
```

---

## DICT

Flat sequence of null-terminated UTF-8 key=value pairs.

```
"scene_id=hellscape_01\0"
"step=0\0"
"timestamp=2026-03-30T00:00:00Z\0"
"\0"   ← terminator (empty key ends dict)
```

---

## FIELD MAP

One `FieldDescriptor` per lane, describing the GPU binding.

```c
struct FieldDescriptor {
    char     name[32];       // e.g. "position", "velocity", "signal"
    uint8_t  gpu_register;   // register slot (matches memory_layout.md)
    uint8_t  element_size;   // bytes per element (4 = float, 16 = float4)
    uint8_t  components;     // 1, 2, 3, or 4
    uint8_t  dtype;          // 0 = float32, 1 = uint32, 2 = int32
    uint32_t lane_index;     // index into LANES array
    uint8_t  reserved[12];   // must be zero
};
```

---

## LANES

One `LaneDescriptor` per data lane.

```c
struct LaneDescriptor {
    uint64_t offset;         // byte offset within PAYLOAD
    uint64_t size;           // byte size of this lane
    uint32_t stride;         // bytes per element
    uint32_t count;          // number of elements (must == entity_count)
    uint8_t  compressed;     // 0 = raw, 1 = lz4, 2 = zstd
    uint8_t  reserved[7];    // must be zero
};
```

---

## PAYLOAD

Raw binary data for all lanes, laid out contiguously.

```
[lane 0 data][lane 1 data]...[lane N data]
```

Each lane is aligned to 16 bytes (padding bytes = 0x00).

---

## Standard Lane Names → GPU Buffer Binding

| Lane Name      | GPU Buffer     | dtype   | Components | Register |
|----------------|----------------|---------|------------|----------|
| `entities`     | `entities`     | uint32  | 1          | `t0`     |
| `position`     | `position`     | float32 | 4          | `t1`     |
| `velocity`     | `velocity`     | float32 | 4          | `t2`     |
| `signal`       | `signal`       | float32 | 1          | `t3`     |
| `axes_row0`    | `axes`         | float32 | 4          | `t4`     |
| `axes_row1`    | `axes`         | float32 | 4          | `t4`     |
| `axes_row2`    | `axes`         | float32 | 4          | `t4`     |

Output lanes (`force`, `events`, `event_params`) are not stored in SCXQ2 —
they are GPU-only output buffers, optionally saved in a separate SCXQ2 snapshot.

---

## Loading Flow

```
SCXQ2 file
    │
    ▼
Parse header → validate magic + version
    │
    ▼
Parse DICT → populate metadata map
    │
    ▼
Parse FIELD MAP + LANES → build lane descriptors
    │
    ▼
For each lane:
    Decompress if needed (LZ4/Zstd)
    Copy into D3D12 UPLOAD heap staging buffer
    │
    ▼
CopyBufferRegion → DEFAULT heap GPU buffers
    │
    ▼
ExecuteCommandLists → TransitionBarrier (COPY_DEST → SRV/UAV)
```

---

## Zero-Copy Conditions

Zero-copy (direct map from file into GPU upload buffer) is possible when:

1. Lane is uncompressed (`compressed == 0`)
2. Lane stride matches GPU buffer stride exactly
3. File is mapped with `CreateFileMapping` / `MapViewOfFile`

---

## Versioning

| Version | Date       | Change                        |
|---------|------------|-------------------------------|
| 0.1     | 2026-03-30 | Initial frozen layout         |
