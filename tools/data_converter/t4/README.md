<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# T4 Dataset Converter

Convert a TIER IV T4 dataset into NCore V4 component stores.

## Frame mapping

| T4 | NCore V4 |
|---|---|
| `base_link` | `rig` |
| `map` | `world` |
| sensor channel | sensor id (same string) |

`calibrated_sensor.{translation, rotation}` is stored as a static pose
`sensor -> rig`. `ego_pose` is stored as a dynamic pose `rig -> world`.

## Usage

```bash
python -m tools.data_converter.t4.main \
    --root-dir /path/to/t4_sequence_or_parent_dir \
    --output-dir /path/to/ncore_output \
    t4-v4 [--label-source autolabel|gt-annotation|external]
```

`--root-dir` may point either at a single T4 sequence (a directory containing
`annotation/` and `data/`) or at a parent directory containing multiple such
sequences side-by-side.

### Label source

`sample_annotation.json` may contain either online detector outputs or
human-labeled ground truth depending on how the source dataset was produced.
Set `--label-source` to match (`autolabel` is the default).

## Limitations

- Camera frame intervals are stored as instantaneous (`[ts, ts]`); shutter
  readout duration is not represented.
- Per-ray lidar timestamps default to the scan-start timestamp. To preserve
  per-point times, the upstream rosbag2-to-T4 conversion must be configured to
  include the `time_stamp` field in the point record (`num_lidar_feats: 7`).
- Cuboid annotations follow the nuScenes-style schema. Image-space annotations
  (`object_ann`, `surface_ann`) are not converted.
