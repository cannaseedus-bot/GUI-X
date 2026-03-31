#!/usr/bin/env python3
"""
pack_scxq2.py  —  MX2LM D3D12 Fused Runtime
Pack NumPy arrays into a SCXQ2 binary file.

Usage:
    python pack_scxq2.py output.scxq2 \
        --entities entities.npy \
        --position position.npy \
        --velocity velocity.npy \
        --signal   signal.npy   \
        --axes-row0 axes_r0.npy \
        --axes-row1 axes_r1.npy \
        --axes-row2 axes_r2.npy \
        --meta "scene_id=test_scene_01"

Or generate random data:
    python pack_scxq2.py output.scxq2 --random --entities-count 4096
"""

import struct
import argparse
import sys
import os
import datetime

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ── SCXQ2 constants ───────────────────────────────────────────
MAGIC             = b'SXQ2'
VERSION_MAJOR     = 0
VERSION_MINOR     = 1

DTYPE_FLOAT32     = 0
DTYPE_UINT32      = 1
DTYPE_INT32       = 2

# ── Standard lane definitions ─────────────────────────────────
STANDARD_LANES = [
    # (name, gpu_register, element_size, components, dtype)
    ("entities",  0, 4, 1, DTYPE_UINT32),
    ("position",  1, 16, 4, DTYPE_FLOAT32),
    ("velocity",  2, 16, 4, DTYPE_FLOAT32),
    ("signal",    3, 4,  1, DTYPE_FLOAT32),
    ("axes_row0", 4, 16, 4, DTYPE_FLOAT32),
    ("axes_row1", 4, 16, 4, DTYPE_FLOAT32),
    ("axes_row2", 4, 16, 4, DTYPE_FLOAT32),
]

def align16(n):
    return (n + 15) & ~15

def pack_string(s):
    return s.encode('utf-8') + b'\x00'

def make_header(entity_count, lane_count,
                dict_offset, dict_size,
                fieldmap_offset, lanes_offset,
                payload_offset, payload_size):
    # magic[4], version_major[2], version_minor[2],
    # entity_count[4], lane_count[4],
    # dict_offset[4], dict_size[4],
    # fieldmap_offset[4], lanes_offset[4],
    # payload_offset[8], payload_size[8],
    # reserved[6]
    return struct.pack('<4sHHIIIIIIQQ6s',
        MAGIC, VERSION_MAJOR, VERSION_MINOR,
        entity_count, lane_count,
        dict_offset, dict_size,
        fieldmap_offset, lanes_offset,
        payload_offset, payload_size,
        b'\x00' * 6)

def make_field_descriptor(name, gpu_register, element_size, components, dtype, lane_index):
    name_bytes = name.encode('utf-8')[:31].ljust(32, b'\x00')
    return struct.pack('<32sBBBBI12s',
        name_bytes,
        gpu_register, element_size, components, dtype,
        lane_index,
        b'\x00' * 12)

def make_lane_descriptor(offset, size, stride, count):
    return struct.pack('<QQIIB7s',
        offset, size, stride, count,
        0,           # compressed = 0 (raw)
        b'\x00' * 7)

def pack_scxq2(output_path, lane_arrays, entity_count, meta_dict=None):
    """
    lane_arrays: list of (name, numpy_array) tuples
    meta_dict:   optional dict of string→string metadata
    """
    if not HAS_NUMPY:
        raise ImportError("NumPy required for pack_scxq2")

    # ── Build DICT ────────────────────────────────────────────
    dict_bytes = b''
    if meta_dict:
        for k, v in meta_dict.items():
            dict_bytes += pack_string(f"{k}={v}")
    dict_bytes += b'\x00'  # terminator
    dict_size = len(dict_bytes)

    lane_count = len(lane_arrays)

    # ── Compute offsets ───────────────────────────────────────
    HEADER_SIZE    = 48
    fieldmap_size  = lane_count * 52   # sizeof(FieldDescriptor)
    lanes_size     = lane_count * 32   # sizeof(LaneDescriptor)

    dict_offset     = HEADER_SIZE
    fieldmap_offset = dict_offset + dict_size
    lanes_offset    = fieldmap_offset + fieldmap_size
    payload_offset  = align16(lanes_offset + lanes_size)

    # ── Build payload ─────────────────────────────────────────
    lane_payloads = []
    for _, arr in lane_arrays:
        raw = np.ascontiguousarray(arr).tobytes()
        # Align to 16 bytes
        pad = align16(len(raw)) - len(raw)
        lane_payloads.append(raw + b'\x00' * pad)

    payload_size = sum(len(p) for p in lane_payloads)

    # ── Build lane offsets ────────────────────────────────────
    lane_offsets = []
    cursor = 0
    for p in lane_payloads:
        lane_offsets.append(cursor)
        cursor += len(p)

    # ── Serialize ─────────────────────────────────────────────
    hdr = make_header(entity_count, lane_count,
                      dict_offset, dict_size,
                      fieldmap_offset, lanes_offset,
                      payload_offset, payload_size)

    field_map_bytes = b''
    lane_desc_bytes = b''

    for idx, (name, arr) in enumerate(lane_arrays):
        # Look up standard definition
        std = next((s for s in STANDARD_LANES if s[0] == name), None)
        if std:
            _, gpu_reg, elem_size, components, dtype = std
        else:
            gpu_reg, elem_size, components, dtype = 0, 4, 1, DTYPE_FLOAT32

        field_map_bytes += make_field_descriptor(
            name, gpu_reg, elem_size, components, dtype, idx)

        stride = arr.itemsize * (arr.shape[1] if arr.ndim > 1 else 1)
        count  = arr.shape[0]
        size   = len(lane_payloads[idx])

        lane_desc_bytes += make_lane_descriptor(
            lane_offsets[idx], size, stride, count)

    # Pad between lanes section and payload
    pad_to_payload = payload_offset - (lanes_offset + lanes_size)
    pad_bytes = b'\x00' * pad_to_payload

    with open(output_path, 'wb') as f:
        f.write(hdr)
        f.write(dict_bytes)
        f.write(field_map_bytes)
        f.write(lane_desc_bytes)
        f.write(pad_bytes)
        for p in lane_payloads:
            f.write(p)

    total = os.path.getsize(output_path)
    print(f"[pack_scxq2] Written: {output_path}  ({total:,} bytes)")
    print(f"             Entities: {entity_count}  |  Lanes: {lane_count}")


def generate_random(entity_count):
    rng = np.random.default_rng(42)
    entities  = np.arange(entity_count, dtype=np.uint32)
    position  = np.column_stack([
        rng.uniform(-30, 30, entity_count).astype(np.float32),
        rng.uniform(-30, 30, entity_count).astype(np.float32),
        rng.uniform(-30, 30, entity_count).astype(np.float32),
        np.ones(entity_count, dtype=np.float32)
    ])
    velocity  = np.column_stack([
        rng.uniform(-1, 1, entity_count).astype(np.float32),
        rng.uniform(-1, 1, entity_count).astype(np.float32),
        rng.uniform(-1, 1, entity_count).astype(np.float32),
        np.ones(entity_count, dtype=np.float32)
    ])
    signal    = rng.uniform(0, 8, entity_count).astype(np.float32)
    axes_row0 = np.tile([1, 0, 0, 0], (entity_count, 1)).astype(np.float32)
    axes_row1 = np.tile([0, 1, 0, 0], (entity_count, 1)).astype(np.float32)
    axes_row2 = np.tile([0, 0, 1, 0], (entity_count, 1)).astype(np.float32)

    return [
        ("entities",  entities.reshape(-1, 1)),
        ("position",  position),
        ("velocity",  velocity),
        ("signal",    signal.reshape(-1, 1)),
        ("axes_row0", axes_row0),
        ("axes_row1", axes_row1),
        ("axes_row2", axes_row2),
    ]


# ── CLI ───────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pack SCXQ2 binary format')
    parser.add_argument('output', help='Output .scxq2 path')
    parser.add_argument('--random', action='store_true', help='Generate random data')
    parser.add_argument('--entities-count', type=int, default=4096)
    parser.add_argument('--entities',  help='entities.npy path')
    parser.add_argument('--position',  help='position.npy path')
    parser.add_argument('--velocity',  help='velocity.npy path')
    parser.add_argument('--signal',    help='signal.npy path')
    parser.add_argument('--axes-row0', help='axes_row0.npy path')
    parser.add_argument('--axes-row1', help='axes_row1.npy path')
    parser.add_argument('--axes-row2', help='axes_row2.npy path')
    parser.add_argument('--meta', action='append', default=[],
                        help='key=value metadata (repeat for multiple)')
    args = parser.parse_args()

    if not HAS_NUMPY:
        print('[ERROR] NumPy not installed: pip install numpy')
        sys.exit(1)

    meta = {'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'}
    for m in args.meta:
        if '=' in m:
            k, v = m.split('=', 1)
            meta[k] = v

    if args.random:
        lane_arrays = generate_random(args.entities_count)
        entity_count = args.entities_count
    else:
        lane_arrays = []
        entity_count = None
        for name, attr in [('entities', args.entities), ('position', args.position),
                           ('velocity', args.velocity), ('signal', args.signal),
                           ('axes_row0', args.axes_row0), ('axes_row1', args.axes_row1),
                           ('axes_row2', args.axes_row2)]:
            if attr:
                arr = np.load(attr)
                lane_arrays.append((name, arr))
                if entity_count is None:
                    entity_count = arr.shape[0]

        if not lane_arrays:
            print('[ERROR] No input lanes provided. Use --random or specify .npy files.')
            sys.exit(1)

    pack_scxq2(args.output, lane_arrays, entity_count, meta)
