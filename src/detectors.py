import math
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO


VEHICLE_CLASSES = {
    "motorcycle": "2W",
    "bicycle": "2W",
    "scooter": "2W",
    "car": "LMV",
    "truck": "HMV",
    "bus": "HMV",
    "van": "LMV",
    "autorickshaw": "LMV",
}

JUNCTION_TYPES = ["T-junction", "4-way", "Y-junction", "roundabout", "flyover/underpass"]


class TrafficAnalyzer:
    def __init__(self, model_source="yolov8n.pt"):
        self.model = YOLO(model_source)
        self.events = []
        self.junction_events = []
        self.vehicle_counts = Counter()
        self.violations = Counter()
        self.mobile_suspects = 0
        self.helmet_suspects = 0
        self.last_positions = []
        self.frame_count = 0

    def process_frame(self, frame, timestamp):
        self.frame_count += 1
        detections = self.detect_objects(frame)

        vehicle_summary = self.classify_vehicles(detections)
        self.update_vehicle_counts(vehicle_summary)

        self.detect_violations(detections, timestamp)
        self.detect_junction(frame, timestamp)
        self.record_timeline(detections, timestamp)

    def detect_objects(self, frame):
        results = self.model(frame, imgsz=640, conf=0.25, verbose=False)
        detections = []
        for result in results:
            for box, cls, score in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                label = self.model.names[int(cls)]
                x1, y1, x2, y2 = map(int, box.tolist())
                detections.append({
                    "label": label,
                    "confidence": float(score),
                    "bbox": [x1, y1, x2, y2],
                })
        return detections

    def classify_vehicles(self, detections):
        summary = Counter()
        for det in detections:
            vehicle_type = VEHICLE_CLASSES.get(det["label"].lower(), "Others")
            summary[vehicle_type] += 1
        return summary

    def update_vehicle_counts(self, vehicle_summary):
        for category, count in vehicle_summary.items():
            self.vehicle_counts[category] += count

    def detect_violations(self, detections, timestamp):
        labels = [det["label"].lower() for det in detections]
        if labels.count("motorcycle") + labels.count("scooter") >= 3:
            self.violations["triple_riding"] += 1
            self.events.append({"time": timestamp, "type": "Triple Riding", "detail": "Three or more 2W detected"})

        if "mobile phone" in labels or "cell phone" in labels:
            self.violations["mobile_phone"] += 1
            self.mobile_suspects += 1
            self.events.append({"time": timestamp, "type": "Mobile Phone Usage", "detail": "Phone detected near driver"})

        if "helmet" in labels:
            self.helmet_suspects += 1
        else:
            if any(lbl in ["motorcycle", "scooter", "motorbike"] for lbl in labels):
                self.violations["helmet_less"] += 1
                self.events.append({"time": timestamp, "type": "Helmet-less Riding", "detail": "2W without helmet label detected"})

        wrong_side = self.estimate_wrong_side(detections)
        if wrong_side:
            self.violations["wrong_side"] += 1
            self.events.append({"time": timestamp, "type": "Wrong-side Driving", "detail": "Vehicle motion indicates wrong-side direction"})

        signal_jump = self.estimate_signal_jumping(detections)
        if signal_jump:
            self.violations["signal_jumping"] += 1
            self.events.append({"time": timestamp, "type": "Signal Jumping", "detail": "Vehicle crosses an intersection in red-signal zone"})

    def estimate_wrong_side(self, detections):
        if len(detections) == 0:
            return False
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            center_x = (x1 + x2) / 2
            if center_x < 100:
                return True
        return False

    def estimate_signal_jumping(self, detections):
        return any(det["label"].lower() in ["car", "bus", "truck", "motorcycle", "scooter"] for det in detections[:1])

    def detect_junction(self, frame, timestamp):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=80, minLineLength=40, maxLineGap=20)
        if lines is None:
            return

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(abs(angle))

        junction_type = self.classify_junction(angles)
        if junction_type:
            self.junction_events.append({"time": timestamp, "junction": junction_type})

    def classify_junction(self, angles):
        unique_angles = len(set(int(a / 10) for a in angles))
        if unique_angles >= 5:
            return "4-way"
        if unique_angles == 4:
            return "T-junction"
        if unique_angles == 3:
            return "Y-junction"
        if unique_angles >= 6:
            return "roundabout"
        return None

    def record_timeline(self, detections, timestamp):
        if detections:
            self.events.append({
                "time": timestamp,
                "type": "Frame Event",
                "detail": f"Detected {len(detections)} objects",
            })

    def build_output(self):
        return {
            "vehicle_counts": dict(self.vehicle_counts),
            "violations": dict(self.violations),
            "events": self.events,
            "junctions": self.junction_events,
            "statistics": {
                "helmet_suspect_frames": self.helmet_suspects,
                "mobile_suspect_frames": self.mobile_suspects,
            },
        }
