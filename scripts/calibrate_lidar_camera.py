#!/usr/bin/env python3
"""Estimate T_lidar_to_cam from 3D-2D correspondences (field calibration).

Field procedure (dock/presa, no special hardware besides a tape measure):
  1. Intrinsics: point the webcam at a printed checkerboard and run
       ros2 run camera_calibration cameracalibrator --size 9x7 --square 0.03 \
         --image:=/bebblebrox/video --camera:=/bebblebrox/video
     then SAVE (writes e.g. /tmp/calibrationdata.tar.gz; extract the
     ost.yaml inside) and pass it as --intrinsics.
  2. Boat static. An assistant holds a pole with reflective tape at N>=8
     spots around the boat (5-20 m, spread left/center/right, near/far).
  3. For each spot: tape-measure the pole base position relative to the
     BOAT origin (x fwd, y left/right, z up), add the pole height used as
     the clicked point (tip or tape band), then measure the LIDAR mount
     offset once (lidar xyz in boat frame) and convert:
         P_lidar = P_boat - lidar_offset
  4. Save one image per spot (or one bag) and click the pole tip/band pixel.
  5. Write correspondences TSV:  u_px  v_px  X_lidar  Y_lidar  Z_lidar
  6. Run this script with --intrinsics + --corr. With --fusion-out it
     patches config/fusion.yaml (K, D, size, extrinsic_matrix) in place.

Usage:
    python3 calibrate_lidar_camera.py --intrinsics ost.yaml \
        --corr correspondences.tsv --fusion-out config/fusion.yaml

    # without ROS cameracalibrator output (fallback, old usage):
    python3 calibrate_lidar_camera.py --K fx 0 cx 0 fy cy 0 0 1 \
        --dist 0 0 0 0 0 --corr correspondences.tsv

Output: 4x4 T_lidar_to_cam (stdout + patched fusion.yaml),
per-point reprojection errors and mean. Accept if mean < 3 px.
Conventions: lidar frame x-fwd/y-left/z-up (VLP16), camera pinhole
Z-forward via cv2.solvePnP (object points are given in lidar frame,
so rvec/tvec ARE the lidar->camera transform; no extra inversion).
"""
import argparse
import shutil
import sys

import cv2
import numpy as np
import yaml


def load_corr(path: str):
    pts2, pts3 = [], []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) != 5:
                raise ValueError(f"{path}:{ln}: want 5 cols, got {len(p)}")
            u, v, x, y, z = map(float, p)
            pts2.append([u, v])
            pts3.append([x, y, z])
    if len(pts2) < 6:
        raise ValueError(f"need >=6 correspondences, got {len(pts2)}")
    return np.array(pts2), np.array(pts3)


def load_intrinsics(path: str):
    """Read ROS cameracalibrator ost.yaml -> (K 3x3, D, width, height)."""
    with open(path) as f:
        c = yaml.safe_load(f)
    K = np.array(c["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    D = np.array(c["distortion_coefficients"]["data"], dtype=np.float64)
    return K, D, int(c["image_width"]), int(c["image_height"])


def patch_fusion(path: str, K, D, width: int, height: int, T) -> None:
    """Write K/D/size/extrinsics into fusion.yaml, keeping other keys."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["camera"]["camera_matrix"] = [float(v) for v in K.reshape(-1)]
    cfg["camera"]["dist_coeffs"] = [float(v) for v in D.reshape(-1)]
    cfg["camera"]["image_width"] = width
    cfg["camera"]["image_height"] = height
    cfg["extrinsic_matrix"] = [[float(v) for v in row] for row in T]
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", nargs=9, type=float, default=None)
    ap.add_argument("--dist", nargs="+", type=float, default=[0, 0, 0, 0, 0])
    ap.add_argument("--intrinsics", default=None,
                    help="ost.yaml from ROS cameracalibrator (preferred over --K/--dist)")
    ap.add_argument("--corr", required=True)
    ap.add_argument("--fusion-out", default=None,
                    help="patch this fusion.yaml with K/D/size/T and copy --intrinsics to --caminfo-out")
    ap.add_argument("--caminfo-out", default=None,
                    help="where usb_cam camera_info_url points (receives copy of --intrinsics)")
    args = ap.parse_args()

    if args.intrinsics:
        K, D, width, height = load_intrinsics(args.intrinsics)
    elif args.K:
        K = np.array(args.K, dtype=np.float64).reshape(3, 3)
        D = np.array(args.dist, dtype=np.float64)
        width, height = 0, 0
    else:
        ap.error("need --intrinsics (preferred) or --K/--dist")
    uv, xyz = load_corr(args.corr)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        xyz.reshape(-1, 1, 3), uv.reshape(-1, 1, 2), K, D,
        flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=4.0,
        confidence=0.99, iterationsCount=300)
    if not ok or inliers is None:
        print("solvePnP failed")
        return 1
    inl = set(int(i[0]) for i in inliers)
    # Refine on inliers only
    ok, rvec, tvec = cv2.solvePnP(
        xyz[list(inl)].reshape(-1, 1, 3), uv[list(inl)].reshape(-1, 1, 2),
        K, D, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()

    proj, _ = cv2.projectPoints(xyz.reshape(-1, 1, 3), rvec, tvec, K, D)
    err = np.linalg.norm(proj.reshape(-1, 2) - uv, axis=1)
    print(f"inliers: {len(inl)}/{len(uv)}")
    for i, e in enumerate(err):
        tag = "IN " if i in inl else "OUT"
        print(f"  pt{i:02d} {tag} err={e:5.2f}px uv=({uv[i][0]:7.1f},{uv[i][1]:7.1f})")
    in_err = err[list(inl)]
    print(f"mean inlier err: {in_err.mean():.2f}px  max: {in_err.max():.2f}px")
    print("extrinsic_matrix:")
    for row in T:
        print(" - - " + "\n   - ".join(f"{v:.6f}" for v in row))
    if args.fusion_out:
        if not args.intrinsics:
            print("refusing --fusion-out without --intrinsics (size unknown)")
            return 1
        patch_fusion(args.fusion_out, K, D, width, height, T)
        print(f"patched {args.fusion_out}")
        if args.caminfo_out:
            shutil.copy(args.intrinsics, args.caminfo_out)
            print(f"camera_info -> {args.caminfo_out} (point usb_cam camera_info_url here)")
    return 0 if in_err.mean() < 3.0 else 2


if __name__ == "__main__":
    sys.exit(main())
