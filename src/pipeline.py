import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from detectors import TrafficAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic analysis pipeline for dashcam videos")
    parser.add_argument("--video", required=True, help="Path to input dashcam video")
    parser.add_argument("--output", default="output.json", help="Path to JSON output summary")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model weight file or model name")
    parser.add_argument("--skip_frames", type=int, default=5, help="Process every Nth frame for speed")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)
    output_path = Path(args.output)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    analyzer = TrafficAnalyzer(model_source=args.model)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = 0
    processed = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % args.skip_frames != 0:
            frame_index += 1
            continue

        timestamp = frame_index / fps
        analyzer.process_frame(frame, timestamp)
        frame_index += 1
        processed += 1

    cap.release()

    output = analyzer.build_output()
    output["video"] = str(video_path.name)
    output["processed_frames"] = processed
    output["duration_seconds"] = round(time.time() - start_time, 2)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
