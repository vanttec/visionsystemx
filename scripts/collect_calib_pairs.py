#!/usr/bin/env python3
"""Collect LiDAR-camera correspondences for calibrate_lidar_camera.py.

One click in rviz anywhere on the board -> the board points around it are
extracted (RANSAC plane), its center is estimated, a window shows the camera
frame of that same moment, you click the board center mark, and the line
``u v x y z`` is appended to <out_dir>/corr.tsv (fsync'd) with the frame
saved as <out_dir>/posNN.png.

Board center (``board_mode``):
  diamond  (default) square board held ROTATED 45 deg (a rhombus). Each VLP16
           ring cuts it with chord width w = 2*(a - |h|), a = side/sqrt(2),
           h = ring height above the center, so the center height is solved
           from the widths even when no ring passes through it.
  centroid mean of the board points; vertical error up to half a ring gap.

Usage (Jetson; the apt OpenCV is needed because the CUDA build has no GUI):
    PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH \
      ros2 run visionsystemx collect_calib_pairs.py --ros-args \
      -p board_side:=0.60 -p out_dir:=$HOME/calib_pairs

  rviz: Fixed Frame = velodyne, PointCloud2 /velodyne_points, optionally
  PointCloud2 /calib/board_points and PointStamped /calib/board_center to
  check what was picked. Use "Publish Point" on the board.
  Window: left click = board center pixel, Enter = save, Esc = discard.
  Works the same with `ros2 bag play` (pause the bag while clicking).

Then:
    python3 calibrate_lidar_camera.py --intrinsics <ost.yaml> \
        --corr <out_dir>/corr.tsv --fusion-out config/fusion.yaml

Self-test of the center math (no ROS needed):
    python3 collect_calib_pairs.py --selftest
"""
import os
import sys

import numpy as np


def fit_plane(P, thresh, iters=200, seed=0, through=None):
    """RANSAC plane on (N,3) -> (inlier mask, unit normal).

    ``through``: a point the plane must contain (the rviz click), so a
    denser surface behind the board (person, wall) cannot win.
    """
    rng = np.random.default_rng(seed)
    best = np.zeros(len(P), bool)
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if through is not None and abs((through - a) @ n) > thresh:
            continue
        inl = np.abs((P - a) @ n) < thresh
        if inl.sum() > best.sum():
            best = inl
    # refine on inliers
    c0 = P[best].mean(axis=0)
    n = np.linalg.svd(P[best] - c0)[2][2]
    return np.abs((P - c0) @ n) < thresh, n


def board_center(P, rings, side, mode):
    """Board points (N,3) + ring ids -> (center xyz, info str)."""
    if mode == "centroid" or rings is None:
        return P.mean(axis=0), f"centroid n={len(P)}"
    _, n = fit_plane(P, 0.05)
    c0 = P.mean(axis=0)
    e_v = np.array([0.0, 0.0, 1.0]) - n[2] * n          # lidar up, in plane
    e_v /= np.linalg.norm(e_v)
    e_h = np.cross(e_v, n)
    s, t = (P - c0) @ e_h, (P - c0) @ e_v
    a = side / np.sqrt(2.0)
    mids, tr, hr = [], [], []
    for r in np.unique(rings):
        m = rings == r
        if m.sum() < 3:
            continue
        ss = np.sort(s[m])
        # samples miss each edge by ~half a step on average
        w = ss[-1] - ss[0] + np.median(np.diff(ss))
        mids.append((ss[0] + ss[-1]) / 2)
        tr.append(t[m].mean())
        hr.append(np.clip(a - w / 2, 0.0, a))
    if len(tr) < 2:
        raise ValueError(f"need >=2 rings on the board, got {len(tr)}")
    tr, hr = np.array(tr), np.array(hr)
    grid = np.arange(tr.min() - a, tr.max() + a, 0.001)
    cost = ((np.abs(tr[None] - grid[:, None]) - hr[None]) ** 2).sum(axis=1)
    tc = grid[cost.argmin()]
    rms = np.sqrt(cost.min() / len(tr))
    center = c0 + np.median(mids) * e_h + tc * e_v
    return center, f"diamond rings={len(tr)} rms={rms * 100:.1f}cm"


def selftest():
    """Synthetic VLP16 hits on a 0.6 m rhombus + a person behind it."""
    side, a = 0.6, 0.6 / np.sqrt(2.0)
    for true in ([5.0, 0.3, 0.17], [9.0, -1.2, -0.4], [4.0, 1.0, 0.55]):
        true = np.array(true)
        rng = np.random.default_rng(1)
        pts, rings = [], []
        for ring, el in enumerate(np.deg2rad(np.arange(-15, 16, 2))):
            for az in np.deg2rad(np.arange(-40, 40, 0.2)):
                d = np.array([np.cos(el) * np.cos(az),
                              np.cos(el) * np.sin(az), np.sin(el)])
                p = d * true[0] / d[0]
                if abs(p[1] - true[1]) + abs(p[2] - true[2]) <= a:
                    pts.append(p + rng.normal(0, 0.005, 3))
                    rings.append(ring)
        P = np.array(pts)
        P[:, 0] += rng.normal(0, 0.01, len(P))
        person = true + rng.normal([0.4, 0, -0.3], 0.1, (40, 3))
        P, rings = np.vstack([P, person]), np.array(rings + [99] * 40)
        inl, _ = fit_plane(P, 0.05, through=P[0])  # click lands on a board point
        c, info = board_center(P[inl], rings[inl], side, "diamond")
        err = np.abs(c - true)
        print(f"true={true} est={c.round(3)} err_cm={(err * 100).round(1)} {info}")
        assert err[1] < 0.015 and err[2] < 0.015, "center error > 1.5 cm"
    print("selftest ok")


def main():
    import cv2
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import PointStamped
    from sensor_msgs.msg import Image, PointCloud2
    from sensor_msgs_py import point_cloud2 as pc2

    # not imported from extract_shape_samples: that pulls usv_interfaces,
    # which a plain ROS container (laptop + bag) does not have
    def to_bgr(msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding == "bgr8":
            return arr.reshape(msg.height, msg.width, 3).copy()
        if msg.encoding == "rgb8":
            return cv2.cvtColor(arr.reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2BGR)
        if msg.encoding in ("mono8", "8UC1"):
            return cv2.cvtColor(arr.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
        raise ValueError(f"unsupported encoding {msg.encoding}")

    class Collector(Node):
        def __init__(self):
            super().__init__("collect_calib_pairs")
            self.cloud_topic = self.declare_parameter("cloud_topic", "/velodyne_points").value
            self.image_topic = self.declare_parameter("image_topic", "/bebblebrox/video").value
            self.out_dir = os.path.expanduser(
                self.declare_parameter("out_dir", "~/calib_pairs").value)
            self.side = float(self.declare_parameter("board_side", 0.60).value)
            self.mode = self.declare_parameter("board_mode", "diamond").value
            self.radius = float(self.declare_parameter("radius", 0.8).value)
            os.makedirs(self.out_dir, exist_ok=True)
            self.cloud = self.image = self.pending = None
            self.create_subscription(PointCloud2, self.cloud_topic,
                                     lambda m: setattr(self, "cloud", m), qos_profile_sensor_data)
            self.create_subscription(Image, self.image_topic,
                                     lambda m: setattr(self, "image", m), qos_profile_sensor_data)
            self.create_subscription(PointStamped, "/clicked_point", self._on_click, 10)
            self.pub_pts = self.create_publisher(PointCloud2, "/calib/board_points", 1)
            self.pub_ctr = self.create_publisher(PointStamped, "/calib/board_center", 1)
            self.get_logger().info(
                f"ready: board {self.mode} side={self.side}m -> {self.out_dir}/corr.tsv")

        def _on_click(self, msg):
            if self.cloud is None or self.image is None:
                self.get_logger().warn("no cloud/image received yet")
                return
            if msg.header.frame_id != self.cloud.header.frame_id:
                self.get_logger().warn(
                    f"click in '{msg.header.frame_id}', cloud in "
                    f"'{self.cloud.header.frame_id}': set rviz Fixed Frame to the cloud frame")
                return
            self.pending = (msg, self.cloud, self.image)

        def handle(self):
            click, cloud, image = self.pending
            self.pending = None
            names = [f.name for f in cloud.fields]
            fields = ("x", "y", "z", "ring") if "ring" in names else ("x", "y", "z")
            arr = pc2.read_points(cloud, field_names=fields, skip_nans=True)
            P = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
            rings = arr["ring"].astype(int) if "ring" in fields else None
            q = np.array([click.point.x, click.point.y, click.point.z])
            near = np.linalg.norm(P - q, axis=1) < self.radius
            if near.sum() < 10:
                self.get_logger().warn(f"only {near.sum()} points near the click, skipped")
                return
            inl, _ = fit_plane(P[near], 0.05, through=q)
            B = P[near][inl]
            R = rings[near][inl] if rings is not None else None
            try:
                c, info = board_center(B, R, self.side, self.mode)
            except ValueError as e:
                self.get_logger().warn(f"{e}; move the board closer or use board_mode:=centroid")
                return
            self.pub_pts.publish(pc2.create_cloud_xyz32(cloud.header, B.astype(np.float32)))
            ctr = PointStamped(header=cloud.header)
            ctr.point.x, ctr.point.y, ctr.point.z = map(float, c)
            self.pub_ctr.publish(ctr)
            self.get_logger().info(f"board center {c.round(3)} ({info})")

            img = to_bgr(image)
            uv = self._pick_pixel(img)
            if uv is None:
                self.get_logger().info("discarded")
                return
            idx = len([f for f in os.listdir(self.out_dir) if f.startswith("pos")]) + 1
            cv2.imwrite(os.path.join(self.out_dir, f"pos{idx:02d}.png"), img)
            with open(os.path.join(self.out_dir, "corr.tsv"), "a") as f:
                f.write(f"# pos{idx:02d} {info}\n"
                        f"{uv[0]} {uv[1]} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
                f.flush()
                os.fsync(f.fileno())
            self.get_logger().info(f"saved pos{idx:02d}: {uv[0]} {uv[1]} {c.round(3)}")

        @staticmethod
        def _pick_pixel(img):
            win = "click board center | Enter=save  Esc=discard"
            uv = []
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

            def on_mouse(ev, x, y, *_):
                if ev == cv2.EVENT_LBUTTONDOWN:
                    uv[:] = [x, y]
            cv2.setMouseCallback(win, on_mouse)
            while True:
                show = img.copy()
                if uv:
                    cv2.drawMarker(show, tuple(uv), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
                    cv2.putText(show, f"u={uv[0]} v={uv[1]}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow(win, show)
                k = cv2.waitKey(30) & 0xFF
                if k in (10, 13) and uv:
                    break
                if k == 27:
                    uv = None
                    break
            cv2.destroyWindow(win)
            cv2.waitKey(1)
            return uv

    rclpy.init()
    node = Collector()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.pending:
                node.handle()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
