# Brief Report

## Project Objective

Build an AI-enabled traffic analytics pipeline for Indian dashcam video that detects road violations, identifies junction types, classifies vehicles, and presents insights through a dashboard.

## Approach

1. Use a YOLO-based object detection backbone (`yolov8n`) to detect vehicles and common objects.
2. Apply heuristics to map detected objects into traffic categories:
   - Two-Wheeler (2W)
   - Light Motor Vehicle (LMV)
   - Heavy Motor Vehicle (HMV)
   - Others
3. Use event detection logic to identify candidate violations and junction occurrences.
4. Save structured output in JSON for dashboard consumption.

## Implementation Highlights

- `src/pipeline.py` processes dashcam video frames, invokes detection, and generates `output.json`.
- `src/detectors.py` encapsulates detection, classification, and event logging.
- `app.py` provides an interactive Streamlit dashboard with summary cards, charts, and logs.
- `sample_output.json` shows the required JSON format for results.

## Datasets and Official Sources

For a production submission, use verified sources such as:
- Ministry of Road Transport & Highways (MoRTH) road accident and vehicle statistics
- National Crime Records Bureau (NCRB) Road Accidents data
- Smart City traffic camera pilot data or public government repositories

This report also uses accident-related methodology and performance metrics from the provided PDF `Tasks_1_to_4_Report.pdf`. The detailed summary is available in `reports/official_data_summary.md`.

## Limitations and Next Steps

- Current violation detection uses placeholder heuristics.
- For real-world accuracy, train a custom model for Indian helmet detection, phone usage, and wrong-side motion.
- Add timestamped video overlays and map-based event visualization.
- Integrate official telecom or city traffic APIs for richer junction classification.
