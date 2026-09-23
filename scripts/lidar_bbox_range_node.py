#!/usr/bin/env python3
"""LiDAR-camera frustum association node (ZED replacement, MVP).

Replaces ZED SDK ``ingestCustomBoxObjects + retrieveObjects`` (see
``src/beeblebrox.cpp``) with VLP16 range + monocular bearing:

    ZbboxArray (/yolo/detections, /shapes/detections)
  + PointCloud2 (/velodyne_points)
  + CameraInfo  (/bebblebrox/video/camera_info)
  -> ObjectList (/bebblebrox/objects/yolo, /bebblebrox/objects/shapes)

Per bbox: project cached LiDAR points into the image, keep hits inside the
eroded bbox, take median forward range, reject outliers with MAD, convert
``(u, z)`` to ``x_fwd = z``, ``y_right = (u - cx) * z / fx`` — the same
``LEFT_HANDED_Z_UP`` convention beeblebrox uses today, so missions and
``obstacle_publisher`` need no changes.

Output contract (matches ``objs2markers``): ``ObjectList`` padded to
``pad_to`` entries; invalid objects use ``color=-1, type="ignore", x=y=0``.
``uuid`` from the input bbox is preserved for ByteTrack continuity.
``v_x/v_y`` are 0.0 in the MVP (velocity comes from tracking downstream).

Extrinsic recalibration (mandatory for a new webcam; YAML values are
placeholders from the ZED2i mount):
  1. Calibrate intrinsics: ``ros2 run camera_calibration cameracalibrator``,
     publish them on the CameraInfo topic (preferred, ``use_camera_info``).
  2. Boat static, 10-15 correspondences: LiDAR cluster centroids of
     poles/buoys at 5/10/15 m <-> clicked image pixels.
  3. ``cv2.solvePnP`` -> ``T_lidar_to_cam``, check reprojection < 3 px.
  4. Store in ``config/fusion.yaml``.
"""

import os
from collections import deque

import cv2
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles, QoSProfile, QoSHistoryPolicy

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge

from usv_interfaces.msg import ZbboxArray, ObjectList, Object

from ament_index_python import get_package_share_directory


# --- label maps (must match beeblebrox.cpp map_yolo_label/map_shapes_label) ---
def map_yolo_label(label: int):
    mapping = {
        0: (4, "round"), 1: (2, "round"), 2: (4, "marker"), 3: (1, "round"),
        4: (0, "marker"), 5: (0, "round"), 6: (1, "marker"), 7: (3, "round"),
    }
    return mapping.get(label, (-1, "ignore"))


def map_shapes_label(label: int):
    mapping = {
        0: (2, "circle"), 1: (2, "plus"), 2: (2, "square"),
        3: (2, "triangle"), 4: (3, "duck"), 5: (1, "circle"),
        6: (1, "plus"), 7: (1, "square"), 8: (1, "triangle"),
        9: (0, "circle"), 10: (0, "plus"), 11: (0, "square"),
        12: (0, "triangle"),
    }
    return mapping.get(label, (-1, "ignore"))


def load_fusion_config(name: str = "fusion.yaml") -> dict:
    path = os.path.join(
        get_package_share_directory("visionsystemx"), "config", name
    )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pointcloud2_to_xyz(cloud_msg: PointCloud2) -> np.ndarray:
    """Fast x/y/z extraction, NaN/Inf rows dropped. Returns (N,3) float32."""
    if cloud_msg.height == 0 or cloud_msg.width == 0:
        return np.zeros((0, 3), dtype=np.float32)
    names = [f.name for f in cloud_msg.fields]
    if not all(k in names for k in ("x", "y", "z")):
        return np.zeros((0, 3), dtype=np.float32)
    x_off = next(f.offset for f in cloud_msg.fields if f.name == "x")
    y_off = next(f.offset for f in cloud_msg.fields if f.name == "y")
    z_off = next(f.offset for f in cloud_msg.fields if f.name == "z")
    step = cloud_msg.point_step
    n = cloud_msg.width * cloud_msg.height
    buf = np.frombuffer(cloud_msg.data, dtype=np.uint8)
    try:
        pts = np.lib.stride_tricks.as_strided(
            buf, shape=(n, step), strides=(step, 1)
        )
    except ValueError:
        return np.zeros((0, 3), dtype=np.float32)
    x = pts[:, x_off:x_off + 4].copy().view(dtype=np.float32).reshape(-1)
    y = pts[:, y_off:y_off + 4].copy().view(dtype=np.float32).reshape(-1)
    z = pts[:, z_off:z_off + 4].copy().view(dtype=np.float32).reshape(-1)
    xyz = np.stack([x, y, z], axis=1)
    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid].astype(np.float32)


class LidarBboxRangeNode(Node):
    def __init__(self):
        super().__init__("lidar_bbox_range_node")
        self.declare_parameter("config_file", "fusion.yaml")
        cfg = load_fusion_config(str(self.get_parameter("config_file").value))

        lidar_cfg = cfg.get("lidar", {})
        cam_cfg = cfg.get("camera", {})
        topics = cfg.get("topics", {})
        bbox_cfg = cfg.get("bbox", {})
        self.T_lidar_to_cam = np.array(cfg["extrinsic_matrix"], dtype=np.float64)

        self.lidar_topic = self.declare_parameter(
            "lidar_topic", lidar_cfg.get("lidar_topic", "/velodyne_points")).value
        self.camera_info_topic = self.declare_parameter(
            "camera_info_topic",
            cam_cfg.get("camera_info_topic", "/bebblebrox/video/camera_info")).value
        self.image_topic = self.declare_parameter(
            "image_topic", cam_cfg.get("image_topic", "/bebblebrox/video")).value
        self.yolo_sub_topic = self.declare_parameter(
            "yolo_sub_topic", topics.get("yolo_sub_topic", "/yolo/detections")).value
        self.shapes_sub_topic = self.declare_parameter(
            "shapes_sub_topic", topics.get("shapes_sub_topic", "/shapes/detections")).value
        self.objects_yolo_topic = self.declare_parameter(
            "objects_yolo_topic",
            topics.get("objects_yolo_topic", "/bebblebrox/objects/yolo")).value
        self.objects_shapes_topic = self.declare_parameter(
            "objects_shapes_topic",
            topics.get("objects_shapes_topic", "/bebblebrox/objects/shapes")).value
        self.debug_topic = self.declare_parameter(
            "debug_topic", cam_cfg.get("debug_topic", "/bebblebrox/debug/frustum")).value

        self.min_range = float(self.declare_parameter(
            "min_range", float(lidar_cfg.get("min_range", 0.5))).value)
        self.max_range = float(self.declare_parameter(
            "max_range", float(lidar_cfg.get("max_range", 60.0))).value)
        self.min_hits = int(self.declare_parameter(
            "min_hits", int(lidar_cfg.get("min_hits", 3))).value)
        self.cloud_timeout = float(self.declare_parameter(
            "cloud_timeout", float(lidar_cfg.get("cloud_timeout", 1.0))).value)
        self.accum_scans = int(self.declare_parameter(
            "accumulation_scans", int(lidar_cfg.get("accumulation_scans", 3))).value)
        self.erosion = float(self.declare_parameter(
            "erosion_ratio", float(bbox_cfg.get("erosion_ratio", 0.10))).value)
        self.pad_to = int(self.declare_parameter(
            "pad_to", int(bbox_cfg.get("pad_to", 10))).value)
        self.mad_scale = float(self.declare_parameter(
            "mad_scale", float(bbox_cfg.get("mad_scale", 2.0))).value)
        self.use_camera_info = bool(self.declare_parameter(
            "use_camera_info", bool(cam_cfg.get("use_camera_info", True))).value)
        self.enable_debug = bool(self.declare_parameter("enable_debug", True).value)

        k = np.array(cam_cfg.get("camera_matrix",
                                 [700.0, 0.0, 640.0, 0.0, 700.0, 360.0, 0.0, 0.0, 1.0]),
                      dtype=np.float64).reshape(3, 3)
        d = np.array(cam_cfg.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]),
                      dtype=np.float64)
        self.yaml_K, self.yaml_D = k, d
        self.K, self.D = k.copy(), d.copy()
        self.img_w = int(cam_cfg.get("image_width", 1280))
        self.img_h = int(cam_cfg.get("image_height", 720))
        self.got_camera_info = False

        self.cloud_buf: deque = deque(maxlen=max(1, self.accum_scans))
        self.cloud_time = None
        self._logged_first_cloud = False
        self.latest_image = None
        self.bridge = CvBridge()

        qos = QoSPresetProfiles.SYSTEM_DEFAULT.value
        # NOTE: lidar/image use explicit KeepLast(10): SYSTEM_DEFAULT did not
        # match the ros_gz_bridge publisher QoS in testing (cloud never cached).
        sensor_qos = QoSProfile(depth=10, history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(PointCloud2, self.lidar_topic,
                                 self._on_cloud, sensor_qos)
        self.create_subscription(CameraInfo, self.camera_info_topic,
                                 self._on_camera_info, sensor_qos)
        if self.enable_debug:
            self.create_subscription(Image, self.image_topic,
                                     self._on_image, sensor_qos)
        self.create_subscription(ZbboxArray, self.yolo_sub_topic,
                                 lambda m: self._on_dets(m, "yolo"), qos)
        self.create_subscription(ZbboxArray, self.shapes_sub_topic,
                                 lambda m: self._on_dets(m, "shapes"), qos)

        self.yolo_pub = self.create_publisher(ObjectList, self.objects_yolo_topic, 10)
        self.shapes_pub = self.create_publisher(ObjectList, self.objects_shapes_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_topic, 1)

        self.get_logger().info(
            f"lidar_bbox_range: lidar={self.lidar_topic} caminfo={self.camera_info_topic} "
            f"yolo {self.yolo_sub_topic}->{self.objects_yolo_topic}, "
            f"shapes {self.shapes_sub_topic}->{self.objects_shapes_topic}")

    # --- subscribers ---
    def _on_cloud(self, msg: PointCloud2):
        xyz = pointcloud2_to_xyz(msg)
        if xyz.shape[0] == 0:
            return
        # NOTE: arrival time, not header.stamp. Gazebo bridges stamp with
        # sim time while this node runs on wall clock; comparing header
        # stamps would report ages of decades and discard every cloud.
        # Arrival-based freshness is valid for Velodyne drivers too
        # (driver stamps with system time at capture).
        t = self.get_clock().now().to_msg()
        self.cloud_buf.append(xyz)
        self.cloud_time = t
        if not self._logged_first_cloud:
            self._logged_first_cloud = True
            self.get_logger().info(
                f"first lidar cloud cached: {xyz.shape[0]} pts "
                f"(buffered scans: {len(self.cloud_buf)})")

    def _on_camera_info(self, msg: CameraInfo):
        if self.use_camera_info:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)
            self.img_w, self.img_h = int(msg.width), int(msg.height)
            self.got_camera_info = True

    def _on_image(self, msg: Image):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            pass

    # --- core ---
    def _project_cached(self):
        """Transform + project buffered clouds. Returns (uv (M,2), z_cam (M,))."""
        if not self.cloud_buf:
            return None, None
        xyz_lidar = np.concatenate(list(self.cloud_buf), axis=0).astype(np.float64)
        n = xyz_lidar.shape[0]
        homo = np.hstack([xyz_lidar, np.ones((n, 1))])
        xyz_cam = (homo @ self.T_lidar_to_cam.T)[:, :3]
        front = xyz_cam[:, 2] > 0.05
        xyz_cam = xyz_cam[front]
        if xyz_cam.shape[0] == 0:
            return None, None
        in_range = (xyz_cam[:, 2] >= self.min_range) & (xyz_cam[:, 2] <= self.max_range)
        xyz_cam = xyz_cam[in_range]
        if xyz_cam.shape[0] == 0:
            return None, None
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        uv, _ = cv2.projectPoints(xyz_cam.reshape(-1, 1, 3), rvec, tvec, self.K, self.D)
        return uv.reshape(-1, 2), xyz_cam[:, 2]

    @staticmethod
    def _erode(x0, y0, x1, y1, ratio):
        w, h = x1 - x0, y1 - y0
        dx, dy = w * ratio / 2.0, h * ratio / 2.0
        return x0 + dx, y0 + dy, x1 - dx, y1 - dy

    def _range_for_bbox(self, uv, z_cam, x0, y0, x1, y1):
        x0e, y0e, x1e, y1e = self._erode(x0, y0, x1, y1, self.erosion)
        if x1e <= x0e or y1e <= y0e:
            return None, 0
        inside = (uv[:, 0] >= x0e) & (uv[:, 0] <= x1e) & (uv[:, 1] >= y0e) & (uv[:, 1] <= y1e)
        zs = z_cam[inside]
        if zs.shape[0] < self.min_hits:
            return None, int(zs.shape[0])
        med = float(np.median(zs))
        mad = float(np.median(np.abs(zs - med)) + 1e-6)
        keep = np.abs(zs - med) <= self.mad_scale * mad
        zs = zs[keep]
        if zs.shape[0] < self.min_hits:
            return None, int(zs.shape[0])
        return float(np.median(zs)), int(zs.shape[0])

    def _ignores_for(self, msg: ZbboxArray) -> ObjectList:
        """Per-box ignore entries preserving uuids (used when ranging
        is impossible: no/stale cloud, projection failure)."""
        dets = ObjectList()
        for box in msg.boxes:
            obj = Object()
            obj.color = -1
            obj.type = "ignore"
            obj.uuid = box.uuid
            obj.x, obj.y, obj.v_x, obj.v_y = 0.0, 0.0, 0.0, 0.0
            dets.obj_list.append(obj)
        return dets

    def _on_dets(self, msg: ZbboxArray, kind: str):
        mapper = map_yolo_label if kind == "yolo" else map_shapes_label
        pub = self.yolo_pub if kind == "yolo" else self.shapes_pub

        if self.cloud_time is None:
            self.get_logger().debug(f"[{kind}] no lidar cloud yet")
            self._publish_padded(pub, self._ignores_for(msg))
            return
        now = self.get_clock().now()
        age = (now - rclpy.time.Time.from_msg(self.cloud_time)).nanoseconds * 1e-9
        if age > self.cloud_timeout:
            self.get_logger().debug(f"[{kind}] stale cloud ({age:.2f}s)")
            self._publish_padded(pub, self._ignores_for(msg))
            return

        proj = self._project_cached()
        if proj[0] is None:
            self.get_logger().debug(f"[{kind}] projection empty (no front points)")
            self._publish_padded(pub, self._ignores_for(msg))
            return
        uv, z_cam = proj
        dets = ObjectList()
        fx, cx = float(self.K[0, 0]), float(self.K[0, 2])
        dbg_hits = []

        for box in msg.boxes:
            color, otype = mapper(int(box.label))
            obj = Object()
            obj.color = int(color)
            obj.type = otype
            obj.uuid = box.uuid
            obj.v_x, obj.v_y = 0.0, 0.0
            z_med, nhits = self._range_for_bbox(
                uv, z_cam, box.x0, box.y0, box.x1, box.y1)
            dbg_hits.append(nhits)
            self.get_logger().debug(
                f"[{kind}] box {box.x0},{box.y0},{box.x1},{box.y1} "
                f"uuid={box.uuid} hits={nhits} nproj={len(z_cam)}")
            if z_med is None:
                obj.x, obj.y = 0.0, 0.0
                obj.type = "ignore"
                obj.color = -1
            else:
                u_c = (box.x0 + box.x1) / 2.0
                obj.x = float(z_med)            # forward
                obj.y = float((u_c - cx) * z_med / fx)  # right
                if not (np.isfinite(obj.x) and np.isfinite(obj.y)):
                    obj.x, obj.y = 0.0, 0.0
                    obj.type = "ignore"
                    obj.color = -1
            dets.obj_list.append(obj)

        self._publish_padded(pub, dets)
        if self.enable_debug:
            self._publish_debug(msg, dbg_hits)

    def _publish_padded(self, pub, dets: ObjectList):
        while len(dets.obj_list) < self.pad_to:
            pad = Object()
            pad.color = -1
            pad.x, pad.y, pad.v_x, pad.v_y = 0.0, 0.0, 0.0, 0.0
            pad.type = "ignore"
            pad.uuid = ""
            dets.obj_list.append(pad)
        pub.publish(dets)

    def _publish_debug(self, msg: ZbboxArray, hits):
        if self.latest_image is None:
            return
        img = self.latest_image.copy()
        for box, nh in zip(msg.boxes, hits):
            color = (0, 255, 0) if nh >= self.min_hits else (0, 0, 255)
            cv2.rectangle(img, (box.x0, box.y0), (box.x1, box.y1), color, 2)
            cv2.putText(img, f"{nh}", (box.x0, max(0, box.y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(img, encoding="bgr8"))
        except Exception as e:
            self.get_logger().debug(f"debug publish failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = LidarBboxRangeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
