#!/usr/bin/env python3
"""Collect LiDAR-camera correspondences for calibrate_lidar_camera.py.

One click in rviz anywhere on the board -> the board points around it are
extracted (RANSAC plane), its center is estimated, a window shows the camera
frame of that same moment, you click the board center mark, and the line
``u v x y z`` is appended to <out_dir>/corr.tsv (fsync'd) with the frame
saved as <out_dir>/posNN.png.

Board center (``board_mode``):
  edges    (default) rectangular board (board_w x board_h) held ROTATED,
           roughly 45 deg. Each VLP16 ring cuts it in a chord whose two ends
           depend on the ring height, so center and in-plane rotation are
           fitted to the chord ends of all rings, even when no ring passes
           through the center. Needs >= 2 rings on the board.
  centroid mean of the board points; vertical error up to half a ring gap.

Usage (Jetson; the apt OpenCV is needed because the CUDA build has no GUI):
    PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH \
      ros2 run visionsystemx collect_calib_pairs.py --ros-args \
      -p board_w:=0.70 -p board_h:=0.61 -p out_dir:=$HOME/calib_pairs
  image_topic is a CompressedImage topic (default /bebblebrox/video/compressed;
  usb_cam run by hand publishes /image_raw/compressed). No republish needed.

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


def chord(t, A, B, phi):
    """Ends [lo, hi] of the horizontal line at height t (from the center)
    inside a rectangle of half sizes A, B rotated by phi in its plane."""
    c, s = np.cos(phi), np.sin(phi)
    a1, a2 = (-A - t * s) / c, (A - t * s) / c
    b1, b2 = (t * c - B) / s, (t * c + B) / s
    lo = np.maximum(np.minimum(a1, a2), np.minimum(b1, b2))
    hi = np.minimum(np.maximum(a1, a2), np.maximum(b1, b2))
    return lo, hi


def board_center(P, rings, board_w, board_h, mode, click=None):
    """Board points (N,3) + ring ids -> (center xyz, info str).

    The holder's torso/legs often lie in the board plane right above or
    below it, so only runs of consecutive rings that fit within the board
    diagonal (and contain the click) are fitted; the run with the most rings
    that matches the rectangle wins.
    """
    if mode == "centroid" or rings is None:
        return P.mean(axis=0), f"centroid n={len(P)}"
    _, n = fit_plane(P, 0.05)
    c0 = P.mean(axis=0)
    e_v = np.array([0.0, 0.0, 1.0]) - n[2] * n          # lidar up, in plane
    e_v /= np.linalg.norm(e_v)
    e_h = np.cross(e_v, n)
    s, t = (P - c0) @ e_h, (P - c0) @ e_v
    lo_o, hi_o, tr = [], [], []
    for r in np.unique(rings):
        m = rings == r
        if m.sum() < 3:
            continue
        ss = np.sort(s[m])
        half = np.median(np.diff(ss)) / 2  # samples miss each edge by ~half a step
        lo_o.append(ss[0] - half)
        hi_o.append(ss[-1] + half)
        tr.append(t[m].mean())
    if len(tr) < 2:
        raise ValueError(f"need >=2 rings on the board, got {len(tr)}")
    order = np.argsort(tr)
    lo_o, hi_o, tr = (np.array(v)[order] for v in (lo_o, hi_o, tr))
    A, B = board_w / 2, board_h / 2

    def search(lo_o, hi_o, tr, phis, scs, tcs):
        ph, sc, tc = np.meshgrid(phis, scs, tcs, indexing="ij")
        lo, hi = chord(tr - tc[..., None], A, B, ph[..., None])
        lo, hi = lo + sc[..., None], hi + sc[..., None]
        cost = np.where(hi > lo, (lo - lo_o) ** 2 + (hi - hi_o) ** 2, 1.0).sum(-1)
        i = np.unravel_index(cost.argmin(), cost.shape)
        return phis[i[0]], scs[i[1]], tcs[i[2]], cost[i]

    def fit(lo_o, hi_o, tr):
        # coarse then fine grid; skip near-axis-aligned holds (vertical center
        # is unobservable there: every ring sees the same full width)
        mid = (lo_o + hi_o) / 2
        phis = np.deg2rad(np.r_[np.arange(10, 80.1, 2.5), np.arange(100, 170.1, 2.5)])
        phi, sc, tc, _ = search(lo_o, hi_o, tr, phis,
                                np.arange(mid.min() - 0.4, mid.max() + 0.4, 0.02),
                                np.arange(tr.min() - 0.5, tr.max() + 0.5, 0.02))
        phi, sc, tc, cost = search(lo_o, hi_o, tr,
                                   phi + np.deg2rad(np.arange(-2.5, 2.6, 0.25)),
                                   sc + np.arange(-0.03, 0.031, 0.002),
                                   tc + np.arange(-0.03, 0.031, 0.002))
        return phi, sc, tc, np.sqrt(cost / (2 * len(tr)))

    t_click = None if click is None else (np.asarray(click) - c0) @ e_v
    diag = np.hypot(board_w, board_h)
    best = None
    for i in range(len(tr)):
        for j in range(i + 1, len(tr)):
            if tr[j] - tr[i] > diag:
                break
            if t_click is not None and not tr[i] - 0.06 <= t_click <= tr[j] + 0.06:
                continue
            k = slice(i, j + 1)
            phi, sc, tc, rms = fit(lo_o[k], hi_o[k], tr[k])
            # most rings among good fits (rms < 2.5 cm), else lowest rms
            key = (rms < 0.025, j - i + 1 if rms < 0.025 else -rms)
            if best is None or key > best[0]:
                best = (key, phi, sc, tc, rms, j - i + 1)
    if best is None:
        raise ValueError("no ring run around the click fits the board")
    _, phi, sc, tc, rms, used = best
    center = c0 + sc * e_h + tc * e_v
    return center, (f"edges rings={used}/{len(tr)} rot={np.rad2deg(phi) % 180:.0f}deg "
                    f"rms={rms * 100:.1f}cm")


def selftest():
    """Synthetic VLP16 hits on a rotated 0.70x0.61 board + a person behind,
    and (torso=True) a torso in the board plane right above it."""
    w, h = 0.70, 0.61
    top = np.hypot(w, h) / 2
    for true, rot, torso in (([5.0, 0.3, 0.17], 45, False), ([9.0, -1.2, -0.4], 38, False),
                             ([4.0, 1.0, 0.55], 130, False), ([7.0, 0.0, 0.0], 55, False),
                             ([3.0, 0.15, -0.45], 45, True), ([5.0, -0.7, -0.4], 50, True)):
        true = np.array(true)
        c, s_ = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
        rng = np.random.default_rng(1)
        pts, rings = [], []
        for ring, el in enumerate(np.deg2rad(np.arange(-15, 16, 2))):
            for az in np.deg2rad(np.arange(-40, 40, 0.2)):
                d = np.array([np.cos(el) * np.cos(az),
                              np.cos(el) * np.sin(az), np.sin(el)])
                p = d * true[0] / d[0]
                u, v = p[1] - true[1], p[2] - true[2]
                on_board = abs(u * c + v * s_) <= w / 2 and abs(-u * s_ + v * c) <= h / 2
                on_torso = torso and abs(u) <= 0.2 and top - 0.1 < v < top + 0.6
                if on_board or on_torso:
                    pts.append(p + rng.normal(0, 0.005, 3))
                    rings.append(ring)
        P = np.array(pts)
        P[:, 0] += rng.normal(0, 0.01, len(P))
        person = true + rng.normal([0.4, 0, -0.3], 0.1, (40, 3))
        P, rings = np.vstack([P, person]), np.array(rings + [99] * 40)
        inl, _ = fit_plane(P, 0.05, through=true)  # click on the board
        c, info = board_center(P[inl], rings[inl], w, h, "edges", click=true)
        err = np.abs(c - true)
        print(f"true={true} rot={rot} torso={torso} est={c.round(3)} "
              f"err_cm={(err * 100).round(1)} {info}")
        assert err[1] < 0.015 and err[2] < 0.015, "center error > 1.5 cm"
    print("selftest ok")


def main():
    from collections import deque

    import cv2
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PointStamped
    from sensor_msgs.msg import CompressedImage, PointCloud2
    from sensor_msgs_py import point_cloud2 as pc2

    import time

    class Collector(Node):
        def __init__(self):
            super().__init__("collect_calib_pairs")
            self.cloud_topic = self.declare_parameter("cloud_topic", "/velodyne_points").value
            # compressed + reliable: raw 1280x720 frames over best-effort DDS
            # get dropped, leaving a stale frame that no longer matches the cloud
            self.image_topic = self.declare_parameter(
                "image_topic", "/bebblebrox/video/compressed").value
            self.out_dir = os.path.expanduser(
                self.declare_parameter("out_dir", "~/calib_pairs").value)
            self.board_w = float(self.declare_parameter("board_w", 0.70).value)
            self.board_h = float(self.declare_parameter("board_h", 0.61).value)
            self.mode = self.declare_parameter("board_mode", "edges").value
            self.radius = float(self.declare_parameter("radius", 0.8).value)
            os.makedirs(self.out_dir, exist_ok=True)
            # matched by arrival time, not header stamp: usb_cam stamps lag
            # ~1 s behind the cloud's; bag play reproduces arrival timing
            self.cloud = self.pending = None
            self.images = deque(maxlen=90)  # (arrival, msg), ~3 s of frames
            self.create_subscription(
                PointCloud2, self.cloud_topic,
                lambda m: setattr(self, "cloud", (time.monotonic(), m)), 10)
            self.create_subscription(
                CompressedImage, self.image_topic,
                lambda m: self.images.append((time.monotonic(), m)), 10)
            self.create_subscription(PointStamped, "/clicked_point", self._on_click, 10)
            self.pub_pts = self.create_publisher(PointCloud2, "/calib/board_points", 1)
            self.pub_ctr = self.create_publisher(PointStamped, "/calib/board_center", 1)
            self.get_logger().info(
                f"ready: board {self.mode} {self.board_w}x{self.board_h}m -> "
                f"{self.out_dir}/corr.tsv")

        def _on_click(self, msg):
            if self.cloud is None or not self.images:
                self.get_logger().warn(
                    f"no cloud/image received yet (image_topic={self.image_topic})")
                return
            t_cloud, cloud = self.cloud
            if msg.header.frame_id != cloud.header.frame_id:
                self.get_logger().warn(
                    f"click in '{msg.header.frame_id}', cloud in "
                    f"'{cloud.header.frame_id}': set rviz Fixed Frame to the cloud frame")
                return
            t_img, image = min(self.images, key=lambda e: abs(e[0] - t_cloud))
            if abs(t_img - t_cloud) > 0.1:
                self.get_logger().warn(f"closest image is {t_img - t_cloud:+.2f}s from the cloud")
            self.pending = (msg, cloud, image)

        def process_pending(self):  # not "handle": rclpy Node uses that name
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
                c, info = board_center(B, R, self.board_w, self.board_h, self.mode, click=q)
            except ValueError as e:
                self.get_logger().warn(f"{e}; move the board closer or use board_mode:=centroid")
                return
            self.pub_pts.publish(pc2.create_cloud_xyz32(cloud.header, B.astype(np.float32)))
            ctr = PointStamped(header=cloud.header)
            ctr.point.x, ctr.point.y, ctr.point.z = map(float, c)
            self.pub_ctr.publish(ctr)
            self.get_logger().info(f"board center {c.round(3)} ({info})")

            img = cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR)
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
                node.process_pending()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
