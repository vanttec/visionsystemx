#!/usr/bin/env python3
"""Export an Ultralytics YOLO26 detection checkpoint for VisionSystemX."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a YOLO26 detection checkpoint with embedded NMS"
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=Path,
        required=True,
        help="YOLO26 PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--format",
        choices=("engine", "onnx"),
        default="engine",
        help="Export format (default: engine)",
    )
    parser.add_argument(
        "--input-shape",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=(640, 640),
        help="Static model input size (default: 640 640)",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Ultralytics export device; TensorRT requires a CUDA device (default: 0)",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=None,
        help="TensorRT workspace in GiB (uses the Ultralytics default when omitted)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset (uses the Ultralytics default when omitted)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify the exported graph",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Minimum confidence accepted by embedded NMS (default: 0.25)",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.50,
        help="IoU threshold used by embedded NMS (default: 0.50)",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum detections returned per image (default: 300)",
    )
    return parser.parse_args()


def main(args):
    if args.weights.suffix.lower() != ".pt":
        raise ValueError("--weights must point to a YOLO26 .pt checkpoint")
    if any(dimension <= 0 for dimension in args.input_shape):
        raise ValueError("--input-shape dimensions must be positive")
    if not 0.0 <= args.conf_thres <= 1.0:
        raise ValueError("--conf-thres must be between 0 and 1")
    if not 0.0 <= args.iou_thres <= 1.0:
        raise ValueError("--iou-thres must be between 0 and 1")
    if args.max_det <= 0:
        raise ValueError("--max-det must be positive")

    model = YOLO(str(args.weights), task="detect")
    detection_head = model.model.model[-1]
    if not hasattr(detection_head, "one2one_cv2") or detection_head.reg_max != 1:
        raise ValueError(
            f"'{args.weights}' does not contain the YOLO26 one-to-one detection head"
        )

    export_args = {
        "format": args.format,
        "imgsz": list(args.input_shape),
        "device": args.device,
        "dynamic": False,
        "nms": True,
        "conf": args.conf_thres,
        "iou": args.iou_thres,
        "max_det": args.max_det,
        # Quantization is intentionally omitted: the C++ runtime uses FP32 I/O.
        "simplify": args.simplify,
    }
    if args.workspace is not None:
        export_args["workspace"] = args.workspace
    if args.opset is not None:
        export_args["opset"] = args.opset

    exported_path = model.export(**export_args)
    print(f"YOLO26 {args.format} export saved to {exported_path}")
    print("Expected detection output: [batch, max_det, 6] = [x1, y1, x2, y2, confidence, class_id]")


if __name__ == "__main__":
    main(parse_args())
