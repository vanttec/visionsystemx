#!/usr/bin/env python3
"""Extract sample image crops per shapes label from a rosbag.

The shapes detector's label table is not documented in-repo. This script
dumps, for every distinct label seen on <shapes_topic>, up to K crops of
<image_topic> expanded by <context> so a human can identify color+shape
per label and fill docs (then map_shapes_label).

Usage (inside ROS env):
    python3 extract_shape_samples.py <bag_uri> <out_dir>
        [--image-topic /bebblebrox/video] [--shapes-topic /shapes/detections]
        [--per-label 8] [--context 3.0]

Output: out_dir/label_<L>/img_<i>.png + out_dir/summary.txt
Assumes image encodings bgr8/rgb8/mono8 (bag0/bag1 are bgr8 1280x720).
"""
import argparse
import os
import sys

import cv2
import numpy as np

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from usv_interfaces.msg import ZbboxArray


def to_bgr(msg: Image) -> np.ndarray:
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding == "bgr8":
        return arr.reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        return cv2.cvtColor(arr.reshape(msg.height, msg.width, 3),
                            cv2.COLOR_RGB2BGR)
    if msg.encoding in ("mono8", "8UC1"):
        return cv2.cvtColor(arr.reshape(msg.height, msg.width),
                            cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported encoding {msg.encoding}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_uri")
    ap.add_argument("out_dir")
    ap.add_argument("--image-topic", default="/bebblebrox/video")
    ap.add_argument("--shapes-topic", default="/shapes/detections")
    ap.add_argument("--per-label", type=int, default=8)
    ap.add_argument("--context", type=float, default=3.0)
    args = ap.parse_args()

    reader = SequentialReader()
    reader.open(StorageOptions(uri=args.bag_uri, storage_id="sqlite3"),
                ConverterOptions("", ""))
    latest_img = None
    saved = {}
    total_boxes = 0
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == args.image_topic:
            try:
                latest_img = to_bgr(deserialize_message(data, Image))
            except Exception:
                latest_img = None
        elif topic == args.shapes_topic and latest_img is not None:
            try:
                arr = deserialize_message(data, ZbboxArray)
            except Exception:
                continue
            h, w = latest_img.shape[:2]
            for b in arr.boxes:
                total_boxes += 1
                n = saved.get(b.label, 0)
                if n >= args.per_label:
                    continue
                bw, bh = b.x1 - b.x0, b.y1 - b.y0
                if bw < 4 or bh < 4:
                    continue
                cx, cy = (b.x0 + b.x1) / 2.0, (b.y0 + b.y1) / 2.0
                hw, hh = bw * args.context / 2.0, bh * args.context / 2.0
                x0, y0 = max(0, int(cx - hw)), max(0, int(cy - hh))
                x1, y1 = min(w, int(cx + hw)), min(h, int(cy + hh))
                crop = latest_img[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                d = os.path.join(args.out_dir, f"label_{b.label}")
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, f"img_{n:02d}.png"), crop)
                saved[b.label] = n + 1

    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(f"bag: {args.bag_uri}\nboxes seen: {total_boxes}\n")
        for lbl in sorted(saved):
            f.write(f"label {lbl}: {saved[lbl]} samples\n")
    print(f"boxes seen: {total_boxes}, labels: {sorted(saved)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
