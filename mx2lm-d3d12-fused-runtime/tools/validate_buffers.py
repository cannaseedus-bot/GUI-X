#!/usr/bin/env python3
"""
validate_buffers.py  —  MX2LM D3D12 Fused Runtime
Validate two SCXQ2 output snapshots for deterministic replay equality.

Usage:
    # Check for NaN/Inf in a single output snapshot
    python validate_buffers.py check output.scxq2

    # Compare two snapshots for bit-exact equality
    python validate_buffers.py compare snapshot_a.scxq2 snapshot_b.scxq2

    # Print stats for a snapshot
    python validate_buffers.py stats output.scxq2
"""

import sys
import struct
import numpy as np

MAGIC = b'SXQ2'

# ── SCXQ2 minimal reader ──────────────────────────────────────

def read_header(data):
    hdr = struct.unpack_from('<4sHHIIIIIIQQ6s', data, 0)
    return {
        'magic': hdr[0], 'ver_maj': hdr[1], 'ver_min': hdr[2],
        'entity_count': hdr[3], 'lane_count': hdr[4],
        'dict_offset': hdr[5], 'dict_size': hdr[6],
        'fieldmap_offset': hdr[7], 'lanes_offset': hdr[8],
        'payload_offset': hdr[9], 'payload_size': hdr[10]
    }

def read_field_descriptors(data, offset, count):
    fields = []
    for i in range(count):
        off = offset + i * 52
        name_bytes, gpu_reg, elem_size, comps, dtype, lane_idx, _ = \
            struct.unpack_from('<32sBBBBI12s', data, off)
        fields.append({
            'name': name_bytes.rstrip(b'\x00').decode(),
            'gpu_register': gpu_reg,
            'element_size': elem_size,
            'components': comps,
            'dtype': dtype,
            'lane_index': lane_idx
        })
    return fields

def read_lane_descriptors(data, offset, count):
    lanes = []
    for i in range(count):
        off = offset + i * 32
        loff, lsize, stride, lcount, compressed, _ = \
            struct.unpack_from('<QQIIB7s', data, off)
        lanes.append({
            'offset': loff, 'size': lsize,
            'stride': stride, 'count': lcount,
            'compressed': compressed
        })
    return lanes

def load_scxq2(path):
    with open(path, 'rb') as f:
        data = f.read()

    hdr = read_header(data)
    if hdr['magic'] != MAGIC:
        raise ValueError(f"Invalid SCXQ2 magic in {path}")

    fields = read_field_descriptors(data, hdr['fieldmap_offset'], hdr['lane_count'])
    lanes  = read_lane_descriptors (data, hdr['lanes_offset'],    hdr['lane_count'])

    result = {'header': hdr, 'lanes': {}}
    for f, l in zip(fields, lanes):
        payload_start = hdr['payload_offset'] + l['offset']
        raw = data[payload_start: payload_start + l['size']]

        dtype = np.float32 if f['dtype'] == 0 else \
                np.uint32  if f['dtype'] == 1 else np.int32

        arr = np.frombuffer(raw[:l['count'] * l['stride']], dtype=dtype)
        if f['components'] > 1:
            arr = arr.reshape(l['count'], f['components'])

        result['lanes'][f['name']] = arr

    return result

# ── Commands ──────────────────────────────────────────────────

def cmd_check(path):
    print(f"[check] {path}")
    snap = load_scxq2(path)
    issues = 0

    for name, arr in snap['lanes'].items():
        if arr.dtype == np.float32:
            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
            if nan_count > 0:
                print(f"  [FAIL] {name}: {nan_count} NaN values")
                issues += 1
            if inf_count > 0:
                print(f"  [FAIL] {name}: {inf_count} Inf values")
                issues += 1
            else:
                print(f"  [OK]   {name}: min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")

    if issues == 0:
        print("[OK] No issues found.")
    else:
        print(f"[FAIL] {issues} issue(s) found.")
        sys.exit(1)


def cmd_compare(path_a, path_b):
    print(f"[compare] {path_a}  vs  {path_b}")
    a = load_scxq2(path_a)
    b = load_scxq2(path_b)

    all_match = True
    for name in a['lanes']:
        if name not in b['lanes']:
            print(f"  [SKIP] {name}: not in snapshot B")
            continue

        arr_a = a['lanes'][name]
        arr_b = b['lanes'][name]

        if arr_a.shape != arr_b.shape:
            print(f"  [FAIL] {name}: shape mismatch {arr_a.shape} vs {arr_b.shape}")
            all_match = False
            continue

        if np.array_equal(arr_a, arr_b):
            print(f"  [OK]   {name}: bit-exact match")
        else:
            diff = np.abs(arr_a.astype(np.float64) - arr_b.astype(np.float64))
            print(f"  [FAIL] {name}: max_diff={diff.max():.6e}  mean_diff={diff.mean():.6e}")
            all_match = False

    if all_match:
        print("[OK] Snapshots are deterministically identical.")
    else:
        print("[FAIL] Snapshots differ — determinism violated.")
        sys.exit(1)


def cmd_stats(path):
    print(f"[stats] {path}")
    snap = load_scxq2(path)
    hdr  = snap['header']

    print(f"  Entities : {hdr['entity_count']}")
    print(f"  Lanes    : {hdr['lane_count']}")
    print()

    for name, arr in snap['lanes'].items():
        if arr.dtype == np.float32:
            print(f"  {name:<14} shape={arr.shape}  min={arr.min():+.4f}  max={arr.max():+.4f}"
                  f"  mean={arr.mean():+.4f}  std={arr.std():.4f}")
        else:
            print(f"  {name:<14} shape={arr.shape}  min={arr.min()}  max={arr.max()}")


# ── Entry ─────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == 'check':
        cmd_check(sys.argv[2])
    elif cmd == 'compare':
        if len(sys.argv) < 4:
            print('[ERROR] compare requires two paths')
            sys.exit(1)
        cmd_compare(sys.argv[2], sys.argv[3])
    elif cmd == 'stats':
        cmd_stats(sys.argv[2])
    else:
        print(f'[ERROR] Unknown command: {cmd}')
        sys.exit(1)
