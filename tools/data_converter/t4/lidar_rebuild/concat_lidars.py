#!/usr/bin/env python3
"""Concatenate 6 decoded Hesai lidar clouds into base_link, preserving ring + per-point time.

Reads the per-lidar decoded rosbags (output of nebula offline decode), applies the
two-level static TF chain (base_link -> hesai_<pos>_base_link -> hesai_<pos>) from
tf_static.json, merges all 6 into one cloud per frame, and writes .pcd.bin files
in the layout the T4->NCore converter expects, but WITH real ring + time.

Output point record (float32 x7, matches num_lidar_feats=7 convention):
    x, y, z, intensity, ring, return_type, time_stamp_sec_offset

- ring: per-source channel, offset so each lidar occupies a distinct band
        (top: 0..127, front_left: 1000.., front_right: 2000.., ...)
        — preserves which physical lidar each point came from.
- time_stamp: per-point time as float seconds relative to frame header stamp.
- frames are matched across the 6 lidars by nearest header timestamp to the
  reference (top) lidar, which is the 10Hz master.
"""
from __future__ import annotations
import argparse
import sqlite3, struct, glob, os, json, sys
import numpy as np

# Populated from CLI args in main().
DECODE = ""
OUT = ""
TF: dict = {}

# per-source ring band offset (keeps physical-lidar identity in the ring field)
RING_OFFSET = {
    "top": 0, "front_left": 1000, "front_right": 2000,
    "side_left": 3000, "side_right": 4000, "rear": 5000,
}
POSITIONS = ["top", "front_left", "front_right", "side_left", "side_right", "rear"]


def quat_to_R(x, y, z, w):
    n = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tf_mat(key):
    t = TF[key]
    M = np.eye(4)
    M[:3, :3] = quat_to_R(*t["rotation_xyzw"])
    M[:3, 3] = t["translation"]
    return M


def chain_to_base(pos):
    """Compose base_link <- hesai_<pos> via the two-level chain in tf_static."""
    # find leaf transform  *_base_link -> hesai_<pos>
    leaf = next(k for k, v in TF.items() if v["child"] == f"hesai_{pos}")
    parent = TF[leaf]["parent"]  # hesai_<pos>_base_link
    M = tf_mat(leaf)
    # walk up to base_link
    cur = parent
    chain = [M]
    while cur != "base_link":
        up = next(k for k, v in TF.items() if v["child"] == cur)
        chain.append(tf_mat(up))
        cur = TF[up]["parent"]
    # base_link <- ... <- leaf : multiply parent-most first
    T = np.eye(4)
    for M in reversed(chain):
        T = T @ M
    return T  # maps points in hesai_<pos> frame to base_link


def read_cloud_msgs(pos):
    """Yield (header_stamp_ns, structured_points) for each frame in a lidar's bag."""
    bag = glob.glob(f"{DECODE}/{pos}/**/*.db3", recursive=True)
    if not bag:
        return
    con = sqlite3.connect(bag[0]); cur = con.cursor()
    tid = cur.execute("SELECT id FROM topics WHERE type LIKE '%PointCloud2%'").fetchone()[0]
    for (ts, blob) in cur.execute(f"SELECT timestamp, data FROM messages WHERE topic_id={tid} ORDER BY timestamp"):
        pts = parse_pc2(blob)
        if pts is not None:
            yield ts, pts
    con.close()


def parse_pc2(blob):
    """Parse PointCloud2 CDR blob -> structured array with our 10 fields (point_step=32)."""
    buf = blob[4:]; off = [0]
    def u32():
        v = struct.unpack_from('<I', buf, off[0])[0]; off[0] += 4; return v
    def u8():
        v = struct.unpack_from('<B', buf, off[0])[0]; off[0] += 1; return v
    def al(n):
        r = off[0] % n
        if r: off[0] += (n - r)
    u32(); u32()  # stamp sec, nsec
    sl = u32(); off[0] += sl; al(4)  # frame_id
    h = u32(); w = u32(); al(4); nf = u32()
    for _ in range(nf):
        sl = u32(); off[0] += sl; al(4)
        u32(); u8(); al(4); u32()
    is_be = u8(); al(4)
    point_step = u32(); row_step = u32()
    al(4); data_len = u32()
    npts = h * w
    if npts == 0:
        return None
    raw = np.frombuffer(buf, dtype=np.uint8, count=npts * point_step, offset=off[0]).reshape(npts, point_step)
    dt = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('intensity', 'u1'), ('return_type', 'u1'), ('channel', '<u2'),
        ('azimuth', '<f4'), ('elevation', '<f4'), ('distance', '<f4'),
        ('time_stamp', '<u4'),
    ])
    return raw.view(dt).reshape(-1)


def main():
    global DECODE, OUT, TF
    ap = argparse.ArgumentParser(description="Concatenate 6 decoded lidar clouds into base_link.")
    ap.add_argument("--decoded-dir", required=True, help="Dir with per-lidar decoded rosbags (decode_packets.sh output)")
    ap.add_argument("--tf-static", required=True, help="tf_static.json (export_tf_static.py output)")
    ap.add_argument("--out-dir", required=True, help="Output dir for NNNNN.bin concatenated frames")
    args = ap.parse_args()
    DECODE = args.decoded_dir
    OUT = args.out_dir
    TF = json.load(open(args.tf_static))

    os.makedirs(OUT, exist_ok=True)
    # precompute transforms
    T = {pos: chain_to_base(pos) for pos in POSITIONS}
    print("Transforms to base_link:")
    for pos in POSITIONS:
        print(f"  {pos}: t={T[pos][:3,3].round(3).tolist()}")

    # Load all frames per lidar into memory-lite index (ts -> points)
    # Reference master = top. Match others by nearest header ts within 50ms.
    print("\nLoading top (reference)...")
    top_frames = list(read_cloud_msgs("top"))
    print(f"  top: {len(top_frames)} frames")

    others = {}
    for pos in POSITIONS:
        if pos == "top":
            continue
        frames = list(read_cloud_msgs(pos))
        others[pos] = frames
        print(f"  {pos}: {len(frames)} frames")

    # build ts arrays for matching
    other_ts = {pos: np.array([ts for ts, _ in frames]) for pos, frames in others.items()}

    n_written = 0
    for fi, (top_ts, top_pts) in enumerate(top_frames):
        merged = []
        # top
        merged.append(transform_and_pack(top_pts, T["top"], RING_OFFSET["top"]))
        # match each other lidar by nearest ts
        for pos in POSITIONS:
            if pos == "top":
                continue
            tsarr = other_ts[pos]
            if len(tsarr) == 0:
                continue
            j = int(np.argmin(np.abs(tsarr - top_ts)))
            if abs(int(tsarr[j]) - int(top_ts)) > 50_000_000:  # >50ms, no match
                continue
            merged.append(transform_and_pack(others[pos][j][1], T[pos], RING_OFFSET[pos]))
        cloud = np.concatenate(merged, axis=0)
        cloud.astype(np.float32).tofile(f"{OUT}/{fi:05d}.bin")
        n_written += 1
        if fi % 100 == 0:
            print(f"  frame {fi}: {len(cloud)} pts merged")
    print(f"\nDONE: wrote {n_written} concatenated frames to {OUT}")


def transform_and_pack(pts, T, ring_offset):
    """Transform xyz to base_link, return Nx7 float32: x,y,z,intensity,ring,return_type,time."""
    xyz = np.stack([pts['x'], pts['y'], pts['z']], axis=1).astype(np.float64)
    # drop NaN/inf
    valid = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid]
    p = pts[valid]
    xyz_h = (T[:3, :3] @ xyz.T).T + T[:3, 3]
    out = np.empty((len(xyz_h), 7), dtype=np.float32)
    out[:, 0:3] = xyz_h
    out[:, 3] = p['intensity'].astype(np.float32)
    out[:, 4] = p['channel'].astype(np.float32) + ring_offset
    out[:, 5] = p['return_type'].astype(np.float32)
    # time_stamp is uint32 ns offset within frame -> seconds
    out[:, 6] = p['time_stamp'].astype(np.float32) * 1e-9
    return out


if __name__ == "__main__":
    main()
