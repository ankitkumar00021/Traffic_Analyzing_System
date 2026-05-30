# Official Road Accident Data Summary

This summary is derived from `Tasks_1_to_4_Report.pdf` and captures the Indian road traffic analytics approach and performance metrics for the final submission.

## Overview
- Project: ML/AI Road Traffic Analysis Pipeline
- Focus: Indian dashcam video analytics for traffic violations, junction detection, vehicle classification, and dashboard reporting.
- Models used: YOLOv8x, YOLOv8-Pose, MobileNet-v3, DeepSORT, DeepLabV3+, ResNet50.

## Task 1 — Traffic Violation Detection
- Dataset: Fine-tuned on a curated Indian traffic violation dataset (~18,000 annotated frames).
- Architecture:
  - YOLOv8x for object detection
  - YOLOv8-Pose for helmet and triple-riding detection
  - MobileNet-v3 for phone-usage classification on cropped rider/driver ROIs
- Violation categories:
  - Helmet-less riding
  - Wrong-side driving
  - Signal jumping
  - Mobile phone usage
  - Triple riding
- Performance metrics:
  - Helmet-less Riding: Precision 0.91, Recall 0.87, F1 0.89
  - Wrong-Side Driving: Precision 0.88, Recall 0.82, F1 0.85
  - Signal Jumping: Precision 0.85, Recall 0.79, F1 0.82
  - Mobile Phone Usage: Precision 0.83, Recall 0.76, F1 0.79
  - Triple Riding: Precision 0.90, Recall 0.85, F1 0.87

## Task 2 — Junction & Road Event Detection
- Methodology:
  - Semantic segmentation for road surface and lane markings
  - Optical-flow analysis for traffic stream bifurcation
  - Depth estimation for flyovers/underpasses
  - GPS-map matching for junction confirmation when GPS metadata is available
- Junction types:
  - T-Junction
  - 4-Way / X-Junction
  - Y-Junction
  - Roundabout
  - Flyover / Underpass
- Accuracy:
  - T-Junction: 93%
  - 4-Way / X-Junction: 91%
  - Y-Junction: 86%
  - Roundabout: 89%
  - Flyover / Underpass: 94%

## Task 3 — Vehicle Detection & Classification
- Two-stage pipeline:
  - YOLOv8n for coarse detection
  - ResNet-50 fine-tuned for classification into 2W, LMV, HMV, Others
- Sample count breakdown:
  - Two-Wheeler (2W): 52.4%
  - Light Motor Vehicle (LMV): 33.9%
  - Heavy Motor Vehicle (HMV): 10.7%
  - Others: 3.0%
- Classification accuracy:
  - Two-Wheeler (2W): Precision 0.94, Recall 0.92, F1 0.93
  - LMV: Precision 0.91, Recall 0.89, F1 0.90
  - HMV: Precision 0.95, Recall 0.93, F1 0.94
  - Others: Precision 0.78, Recall 0.74, F1 0.76
  - Macro average: Precision 0.90, Recall 0.87, F1 0.88

## Notes for Submission
- This file documents official task definitions, dataset sources, and model performance metrics.
- Use this summary as the basis for the final report and dashboard documentation.
