#!/usr/bin/env python3
"""Estimate T_lidar_to_cam from 3D-2D correspondences (field calibration).

Field procedure (dock/presa, no special hardware besides a tape measure):
  1. Boat static. An assistant holds a pole with reflective tape at N>=8
     spots around the boat (5-20 m, spread left/center/right, near/far).
  2. For each spot: tape-measure the pole base position relative to the
     BOAT origin (x fwd, y left/right, z up), add the pole height used as
     the clicked point (tip or tape band), then measure the LIDAR mount
     offset once (lidar xyz in boat frame) and convert:
         P_lidar = P_boat - lidar_offset
  3. Save one image per spot (or one bag) and click the pole tip/band pixel.
  4. Write correspondences TSV:  u_px  v_px  X_lidar  Y_lidar  Z_lidar
  5. Run this script with camera intrinsics (from camera_calibration).

Usage:
    python3 calibrate_lidar_camera.py --K fx 0 cx 0 fy cy 0 0 1 \
        --dist 0 0 0 0 0 --corr correspondences.tsv

Output: 4x4 T_lidar_to_cam (row-major YAML block ready for fusion.yaml),
per-point reprojection errors and mean. Accept if mean < 3 px.
Conventions: lidar frame x-fwd/y-left/z-up (VLP16), camera pinhole
Z-forward via cv2.solvePnP (object points are given in lidar frame,
so rvec/tvec ARE the lidar->camera transform; no extra inversion).
"""
import argparse
import sys

import cv2
import numpy as np


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", nargs=9, type=float, required=True)
    ap.add_argument("--dist", nargs="+", type=float, default=[0, 0, 0, 0, 0])
    ap.add_argument("--corr", required=True)
    args = ap.parse_args()

    K = np.array(args.K, dtype=np.float64).reshape(3, 3)
    D = np.array(args.dist, dtype=np.float64)
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
    return 0 if in_err.mean() < 3.0 else 2


if __name__ == "__main__":
    sys.exit(main())
