#!/usr/bin/env python3
"""Export /tf_static from a T4 input_bag .db3 to JSON (sensor extrinsics).

The static transforms give each lidar's pose relative to base_link, needed to
concatenate the decoded per-lidar clouds. Run inside a ROS 2 (Humble) env so
tf2_msgs is importable.

Usage:
    python export_tf_static.py <input_bag.db3> <out.json>
"""
import sqlite3
import json
import sys

from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage


def main(db_path: str, out_path: str) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    tid = cur.execute("SELECT id FROM topics WHERE name=?", ("/tf_static",)).fetchone()[0]
    blob = cur.execute(f"SELECT data FROM messages WHERE topic_id={tid} LIMIT 1").fetchone()[0]
    con.close()

    msg = deserialize_message(blob, TFMessage)
    tf = {}
    for t in msg.transforms:
        p = t.transform.translation
        q = t.transform.rotation
        tf[f"{t.header.frame_id}->{t.child_frame_id}"] = {
            "parent": t.header.frame_id,
            "child": t.child_frame_id,
            "translation": [p.x, p.y, p.z],
            "rotation_xyzw": [q.x, q.y, q.z, q.w],
        }

    with open(out_path, "w") as f:
        json.dump(tf, f, indent=2)
    print(f"saved {len(tf)} transforms to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
