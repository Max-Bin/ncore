<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lidar ring/time rebuild (optional preprocessing)

A T4 dataset's `LIDAR_CONCAT/*.pcd.bin` stores only `x, y, z, intensity` — the
**ring (laser channel) and per-point timestamp are missing** (dropped when the
online point-cloud concatenation runs with `num_lidar_feats=5`, and the
concatenated topic is usually not recorded in the bag). Without per-point time,
lidar motion compensation is impossible.

This submodule recovers real ring + per-point time by decoding the raw Hesai
`pandar_packets` (which carry both at the protocol level) straight from the T4
`input_bag`, then concatenating the per-lidar clouds into `base_link`.

## Requirements

- A built Autoware (Humble) workspace with the `nebula` driver
  (`nebula_examples`, `nebula_decoders`).
- The T4 dataset's `input_bag` (raw `*/pandar_packets` topics present).

## Pipeline

```bash
BAG=<t4_dataset>/input_bag                 # dir containing the .db3
DB3=$BAG/<name>_0.db3                       # the sqlite3 file inside it
OUT=/path/to/lidar_rebuild

# 1. Decode 6 raw lidars -> per-lidar clouds with ring + time
./decode_packets.sh "$BAG" "$OUT/decoded" [autoware_install/setup.bash]

# 2. Export sensor extrinsics from /tf_static (run in ROS 2 env)
python export_tf_static.py "$DB3" "$OUT/tf_static.json"

# 3. Concatenate into base_link (NNNNN.bin, float32 x7)
python concat_lidars.py \
    --decoded-dir "$OUT/decoded" \
    --tf-static "$OUT/tf_static.json" \
    --out-dir "$OUT/concatenated"
```

Then point the T4 converter at the result:

```bash
python -m tools.data_converter.t4.main \
    --root-dir <t4_dataset> --output-dir <ncore_out> \
    t4-v4 --rebuilt-lidar-dir "$OUT/concatenated"
```

## Output format

Each `NNNNN.bin` is `float32` shape `(N, 7)`:
`x, y, z, intensity(0-255), ring, return_type, time_sec_offset`.

`ring` is band-offset per source lidar (top 0–127, front_left 1000+, …) to keep
physical-lidar identity. The converter consumes ring/time but stores the cloud
as **unstructured** (no row/col grid) since the 6-lidar fusion has no single
structured model.

## Sensor models (XX1 Odaiba kit)

Determined from packet-header block counts, not guessed:
- `top` = Pandar128E4X (128-line)
- `front_left/right`, `side_left/right`, `rear` = PandarXT32 (32-line)

Edit `MODELS` in `decode_packets.sh` if your kit differs.
