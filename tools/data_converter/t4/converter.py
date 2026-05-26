# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T4 dataset to NCore V4 converter.

T4 ``base_link`` and ``map`` frames map to NCore ``rig`` and ``world``.
``calibrated_sensor`` stores ``T_sensor_baselink`` and ``ego_pose`` stores
``T_baselink_map``; both match NCore's source->target convention with no
inversion needed.
"""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal

import click
import numpy as np
import tqdm

from upath import UPath

from scipy.spatial.transform import Rotation

from ncore.impl.common.transformations import HalfClosedInterval
from ncore.impl.data.types import (
    BBox3,
    CuboidTrackObservation,
    LabelSource,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from ncore.impl.data.v4.components import (
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PosesComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)
from ncore.impl.data.v4.types import ComponentGroupAssignments
from ncore.impl.data_converter.base import FileBasedDataConverter, FileBasedDataConverterConfig
from tools.data_converter.cli import cli
from tools.data_converter.t4.utils import (
    index_by_token,
    load_annotation_tables,
    load_lidar_xyzi,
    t4_pose_to_se3,
)


T4_LIDAR_INTENSITY_MAX: float = 255.0


@dataclass(kw_only=True, slots=True)
class T4Converter4Config(FileBasedDataConverterConfig):
    """Configuration for T4 to NCore V4 conversion."""

    store_type: Literal["itar", "directory"] = "itar"
    component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
    store_sequence_meta: bool = True
    label_source: Literal["autolabel", "gt-annotation", "external"] = "autolabel"


class T4Converter4(FileBasedDataConverter):
    """T4 dataset to NCore V4 converter."""

    _LABEL_SOURCE_MAP = {
        "autolabel": LabelSource.AUTOLABEL,
        "gt-annotation": LabelSource.GT_ANNOTATION,
        "external": LabelSource.EXTERNAL,
    }

    def __init__(self, config: T4Converter4Config) -> None:
        super().__init__(config)
        self.component_group_profile = config.component_group_profile
        self.store_type = config.store_type
        self.store_sequence_meta = config.store_sequence_meta
        self.label_source = self._LABEL_SOURCE_MAP[config.label_source]
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_sequence_ids(config: T4Converter4Config) -> list[str]:
        """A T4 sequence is a directory holding ``annotation/`` and ``data/``."""
        root = Path(config.root_dir)

        def _is_t4_sequence(p: Path) -> bool:
            return (p / "annotation").is_dir() and (p / "data").is_dir()

        if _is_t4_sequence(root):
            return [str(root)]

        return [str(p) for p in sorted(root.iterdir()) if p.is_dir() and _is_t4_sequence(p)]

    @staticmethod
    def from_config(config: T4Converter4Config) -> T4Converter4:
        return T4Converter4(config)

    def convert_sequence(self, sequence_id: str) -> None:
        sequence_path = Path(sequence_id)
        sequence_name = sequence_path.name
        self.logger.info(f"Converting T4 sequence: {sequence_name}")

        # Lidar scan period drives the per-frame time window. Read from
        # status.json when present; fall back to 100 ms otherwise.
        status_path = sequence_path / "status.json"
        lidar_scan_period_us = 100_000
        if status_path.exists():
            with status_path.open("r") as f:
                status_root = json.load(f)
            status_inner = next(iter(status_root.values())) if status_root else {}
            period_sec = status_inner.get("_lidar_scan_period_sec")
            if period_sec is not None:
                lidar_scan_period_us = int(round(float(period_sec) * 1_000_000))

        tables = load_annotation_tables(sequence_path / "annotation")
        scene = tables["scene"][0]
        sensors_by_token = index_by_token(tables["sensor"])
        calibrated_by_token = index_by_token(tables["calibrated_sensor"])

        sample_data_by_channel: Dict[str, list[dict]] = {}
        for sd in tables["sample_data"]:
            calib = calibrated_by_token[sd["calibrated_sensor_token"]]
            channel = sensors_by_token[calib["sensor_token"]]["channel"]
            sample_data_by_channel.setdefault(channel, []).append(sd)
        for channel in sample_data_by_channel:
            sample_data_by_channel[channel].sort(key=lambda sd: sd["timestamp"])

        camera_channels = sorted(
            ch for ch, sds in sample_data_by_channel.items()
            if sensors_by_token[calibrated_by_token[sds[0]["calibrated_sensor_token"]]["sensor_token"]]["modality"] == "camera"
        )
        lidar_channels = sorted(
            ch for ch, sds in sample_data_by_channel.items()
            if sensors_by_token[calibrated_by_token[sds[0]["calibrated_sensor_token"]]["sensor_token"]]["modality"] == "lidar"
        )

        camera_ids = self.get_active_camera_ids(camera_channels)
        lidar_ids = self.get_active_lidar_ids(lidar_channels)

        # Sequence interval must contain every per-frame window we store.
        sd_ts = [sd["timestamp"] for sd in tables["sample_data"]]
        lidar_ts = [
            sd["timestamp"] for sd in tables["sample_data"]
            if sensors_by_token[calibrated_by_token[sd["calibrated_sensor_token"]]["sensor_token"]]["modality"] == "lidar"
        ]
        seq_start_us = min(sd_ts)
        seq_end_us_inclusive = max(max(sd_ts), max(lidar_ts) + lidar_scan_period_us - 1) if lidar_ts else max(sd_ts)
        sequence_timestamp_interval_us = HalfClosedInterval.from_start_end(seq_start_us, seq_end_us_inclusive)

        ego_records_sorted = sorted(tables["ego_pose"], key=lambda r: r["timestamp"])
        ego_records_dedup: list[dict] = []
        last_ts = None
        for r in ego_records_sorted:
            if r["timestamp"] == last_ts:
                continue
            ego_records_dedup.append(r)
            last_ts = r["timestamp"]

        ego_timestamps_us = np.array([r["timestamp"] for r in ego_records_dedup], dtype=np.uint64)
        T_rig_world = np.stack(
            [t4_pose_to_se3(r["translation"], r["rotation"]) for r in ego_records_dedup],
            axis=0,
        )

        # Replicate boundary poses so the trajectory covers the full interval.
        if ego_timestamps_us[0] > seq_start_us:
            ego_timestamps_us = np.concatenate([np.array([seq_start_us], dtype=np.uint64), ego_timestamps_us])
            T_rig_world = np.concatenate([T_rig_world[:1], T_rig_world], axis=0)
        if ego_timestamps_us[-1] < seq_end_us_inclusive:
            ego_timestamps_us = np.concatenate(
                [ego_timestamps_us, np.array([seq_end_us_inclusive], dtype=np.uint64)]
            )
            T_rig_world = np.concatenate([T_rig_world, T_rig_world[-1:]], axis=0)

        component_groups = ComponentGroupAssignments.create(
            camera_ids=camera_ids,
            lidar_ids=lidar_ids,
            radar_ids=[],
            point_clouds_ids=[],
            camera_labels_ids=[],
            profile=self.component_group_profile,
        )

        store_writer = SequenceComponentGroupsWriter(
            output_dir_path=UPath(self.output_dir) / sequence_name,
            store_base_name=sequence_name,
            sequence_id=sequence_name,
            sequence_timestamp_interval_us=sequence_timestamp_interval_us,
            store_type=self.store_type,
            generic_meta_data={
                "source_format": "t4",
                "t4_scene_token": scene["token"],
                "t4_log_token": scene["log_token"],
            },
        )

        poses_writer = store_writer.register_component_writer(
            PosesComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.poses_component_group,
            generic_meta_data={"calibration_type": "t4:calibrated_sensor", "egomotion_type": "t4:ego_pose"},
        )
        intrinsics_writer = store_writer.register_component_writer(
            IntrinsicsComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.intrinsics_component_group,
        )
        masks_writer = store_writer.register_component_writer(
            MasksComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.masks_component_group,
        )

        poses_writer.store_dynamic_pose(
            source_frame_id="rig",
            target_frame_id="world",
            poses=T_rig_world.astype(np.float32),
            timestamps_us=ego_timestamps_us,
        )

        for lidar_id in lidar_ids:
            self._convert_lidar(
                lidar_id=lidar_id,
                sequence_path=sequence_path,
                sample_data=sample_data_by_channel[lidar_id],
                calibrated_by_token=calibrated_by_token,
                store_writer=store_writer,
                poses_writer=poses_writer,
                component_groups=component_groups,
                scan_period_us=lidar_scan_period_us,
            )

        for camera_id in camera_ids:
            self._convert_camera(
                camera_id=camera_id,
                sequence_path=sequence_path,
                sample_data=sample_data_by_channel[camera_id],
                calibrated_by_token=calibrated_by_token,
                store_writer=store_writer,
                poses_writer=poses_writer,
                intrinsics_writer=intrinsics_writer,
                masks_writer=masks_writer,
                component_groups=component_groups,
            )

        if tables["sample_annotation"]:
            self._convert_cuboids(
                tables=tables,
                store_writer=store_writer,
                component_groups=component_groups,
            )

        ncore_paths = store_writer.finalize()

        if self.store_sequence_meta:
            reader = SequenceComponentGroupsReader(ncore_paths)
            meta_path = UPath(self.output_dir) / sequence_name / f"{reader.sequence_id}.json"
            with meta_path.open("w") as f:
                json.dump(reader.get_sequence_meta().to_dict(), f, indent=2)

    def _convert_lidar(
        self,
        lidar_id: str,
        sequence_path: Path,
        sample_data: list[dict],
        calibrated_by_token: dict,
        store_writer: SequenceComponentGroupsWriter,
        poses_writer: PosesComponent.Writer,
        component_groups: ComponentGroupAssignments,
        scan_period_us: int,
    ) -> None:
        lidar_writer = store_writer.register_component_writer(
            LidarSensorComponent.Writer,
            component_instance_name=lidar_id,
            group_name=component_groups.lidar_component_groups.get(lidar_id),
            generic_meta_data={},
        )

        calib = calibrated_by_token[sample_data[0]["calibrated_sensor_token"]]
        T_sensor_rig = t4_pose_to_se3(calib["translation"], calib["rotation"])
        poses_writer.store_static_pose(
            source_frame_id=lidar_id,
            target_frame_id="rig",
            pose=T_sensor_rig.astype(np.float32),
        )

        for sd in tqdm.tqdm(sample_data, desc=f"lidar {lidar_id}"):
            points = load_lidar_xyzi(sequence_path / sd["filename"])
            xyz = points[:, :3]
            intensity_raw = points[:, 3]

            distance_m = np.linalg.norm(xyz, axis=1).astype(np.float32)
            direction = np.zeros_like(xyz, dtype=np.float32)
            valid = distance_m > 0
            direction[valid] = (xyz[valid] / distance_m[valid, None]).astype(np.float32)

            # NCore requires unit-norm directions; drop zero-distance rays.
            if not valid.all():
                direction = direction[valid]
                distance_m = distance_m[valid]
                intensity_raw = intensity_raw[valid]

            intensity = np.clip(intensity_raw / T4_LIDAR_INTENSITY_MAX, 0.0, 1.0).astype(np.float32)

            ts_us = np.uint64(sd["timestamp"])
            point_timestamps_us = np.full(direction.shape[0], ts_us, dtype=np.uint64)

            frame_start_us = ts_us
            frame_end_us = ts_us + np.uint64(scan_period_us - 1)

            lidar_writer.store_frame(
                direction=direction,
                timestamp_us=point_timestamps_us,
                model_element=None,
                distance_m=distance_m.reshape(1, -1),
                intensity=intensity.reshape(1, -1),
                frame_timestamps_us=np.array([frame_start_us, frame_end_us], dtype=np.uint64),
                generic_data={},
                generic_meta_data={},
            )

    def _convert_camera(
        self,
        camera_id: str,
        sequence_path: Path,
        sample_data: list[dict],
        calibrated_by_token: dict,
        store_writer: SequenceComponentGroupsWriter,
        poses_writer: PosesComponent.Writer,
        intrinsics_writer: IntrinsicsComponent.Writer,
        masks_writer: MasksComponent.Writer,
        component_groups: ComponentGroupAssignments,
    ) -> None:
        calib = calibrated_by_token[sample_data[0]["calibrated_sensor_token"]]
        T_sensor_rig = t4_pose_to_se3(calib["translation"], calib["rotation"])
        poses_writer.store_static_pose(
            source_frame_id=camera_id,
            target_frame_id="rig",
            pose=T_sensor_rig.astype(np.float32),
        )

        K = np.array(calib["camera_intrinsic"], dtype=np.float32)
        fu, fv = float(K[0, 0]), float(K[1, 1])
        cu, cv = float(K[0, 2]), float(K[1, 2])

        width = int(sample_data[0].get("width") or 0)
        height = int(sample_data[0].get("height") or 0)
        if width == 0 or height == 0:
            raise ValueError(f"Camera {camera_id}: sample_data is missing width/height")

        camera_writer = store_writer.register_component_writer(
            CameraSensorComponent.Writer,
            component_instance_name=camera_id,
            group_name=component_groups.camera_component_groups.get(camera_id),
            generic_meta_data={},
        )

        for sd in tqdm.tqdm(sample_data, desc=f"camera {camera_id}"):
            img_path = sequence_path / sd["filename"]
            with img_path.open("rb") as f:
                image_binary = f.read()
            ts_us = np.uint64(sd["timestamp"])
            camera_writer.store_frame(
                image_binary_data=image_binary,
                image_format="jpeg",
                frame_timestamps_us=np.array([ts_us, ts_us], dtype=np.uint64),
                generic_data={},
                generic_meta_data={},
            )

        intrinsics_writer.store_camera_intrinsics(
            camera_id=camera_id,
            camera_model_parameters=OpenCVPinholeCameraModelParameters(
                resolution=np.array([width, height], dtype=np.uint64),
                shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
                external_distortion_parameters=None,
                principal_point=np.array([cu, cv], dtype=np.float32),
                focal_length=np.array([fu, fv], dtype=np.float32),
                radial_coeffs=np.zeros(6, dtype=np.float32),
                tangential_coeffs=np.zeros(2, dtype=np.float32),
                thin_prism_coeffs=np.zeros(4, dtype=np.float32),
            ),
        )
        masks_writer.store_camera_masks(camera_id=camera_id, mask_images={})

    def _convert_cuboids(
        self,
        tables: dict,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
    ) -> None:
        """Convert ``sample_annotation.json`` to CuboidTrackObservations in the world frame.

        T4 follows nuScenes: translation+rotation are in the global frame and
        ``size`` is ``[width, length, height]`` in box-local axes (length along
        local x, width along local y, height along local z). NCore's BBox3.dim
        is ``[dim_x, dim_y, dim_z]`` so we reorder to ``[length, width, height]``.
        """
        sample_ts_by_token = {r["token"]: r["timestamp"] for r in tables["sample"]}
        category_name_by_token = {r["token"]: r["name"] for r in tables["category"]}
        category_token_by_instance = {r["token"]: r["category_token"] for r in tables["instance"]}

        observations: list[CuboidTrackObservation] = []
        for ann in tables["sample_annotation"]:
            sample_ts = sample_ts_by_token[ann["sample_token"]]
            cat_token = category_token_by_instance.get(ann["instance_token"])
            class_id = category_name_by_token.get(cat_token, "unknown")

            tx, ty, tz = ann["translation"]
            width, length, height = ann["size"]
            qw, qx, qy, qz = ann["rotation"]
            rx, ry, rz = Rotation.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)

            observations.append(
                CuboidTrackObservation(
                    track_id=ann["instance_token"],
                    class_id=class_id,
                    timestamp_us=int(sample_ts),
                    reference_frame_id="world",
                    reference_frame_timestamp_us=int(sample_ts),
                    bbox3=BBox3(
                        centroid=(float(tx), float(ty), float(tz)),
                        dim=(float(length), float(width), float(height)),
                        rot=(float(rx), float(ry), float(rz)),
                    ),
                    source=self.label_source,
                )
            )

        store_writer.register_component_writer(
            CuboidsComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.cuboid_track_observations_component_group,
        ).store_observations(observations)


@cli.command()
@click.option(
    "--store-type",
    type=click.Choice(["itar", "directory"], case_sensitive=False),
    default="itar",
    show_default=True,
)
@click.option(
    "component_group_profile",
    "--profile",
    type=click.Choice(["default", "separate-sensors", "separate-all"], case_sensitive=False),
    default="separate-sensors",
    show_default=True,
)
@click.option("store_sequence_meta", "--sequence-meta/--no-sequence-meta", default=True)
@click.option(
    "--label-source",
    type=click.Choice(["autolabel", "gt-annotation", "external"], case_sensitive=False),
    default="autolabel",
    show_default=True,
    help="Provenance to record for cuboids in sample_annotation.json.",
)
@click.pass_context
def t4_v4(ctx, *_, **kwargs):
    """T4 dataset conversion (V4 format)"""
    config = T4Converter4Config(**{**vars(ctx.obj), **kwargs})
    T4Converter4.convert(config)
