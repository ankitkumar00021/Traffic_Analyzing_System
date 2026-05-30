# Traffic Analyzer — Indian Road Dashcam Insights

Bring dashcam video to life: detect violations, classify vehicles, and visualize insights.

## What this project does
- Detects common traffic violations (helmet-less riding, mobile-phone use, wrong-side driving, signal jumping, triple riding)
- Classifies vehicles into Two-Wheeler (2W), Light Motor Vehicle (LMV), Heavy Motor Vehicle (HMV), and Others
- Identifies junction types and road events (T-junction, 4-way, Y-junction, roundabout, flyover/underpass)
- Presents findings in an interactive Streamlit dashboard with charts and timeline logs

## Quick Start (fastest way to run)
1. Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

2. Run the video processing pipeline (uses `dashcam.mp4` in project root by default):

```powershell
.\venv\Scripts\python.exe src\pipeline.py --video .\dashcam.mp4 --output output.json
```

3. Launch the dashboard and click **Load Data** (default `output.json`):

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

> Tip: If the venv is activated, replace `.\venv\Scripts\python.exe` with `python`.

## Project Structure (what's where)
- [app.py](app.py) — Streamlit dashboard (UI + visualization)
- [src/pipeline.py](src/pipeline.py) — Main runner: reads video → runs detectors → writes `output.json`
- [src/detectors.py](src/detectors.py) — Detection, classification, and heuristic analyzers
- [sample_output.json](sample_output.json) — Example JSON for demoing the dashboard
- [output.json](output.json) — Pipeline output file used by the dashboard
- [yolov8n.pt](yolov8n.pt) — Default YOLOv8 weights used for quick tests
- [reports/brief_report.md](reports/brief_report.md) — Short project report and limitations
- [reports/official_data_summary.md](reports/official_data_summary.md) — Extracted accident metrics and model performance from provided report
- [tools/](tools/) — Helper scripts (video generation, inspection utilities)

## Model Performance (official metrics)
We include a compact model performance summary (sourced from the provided report) inside the dashboard. For full details see [reports/official_data_summary.md](reports/official_data_summary.md).

Key validation figures (sample):

- Helmet-less Riding — Precision: 0.91, Recall: 0.87, F1: 0.89
- Wrong-side Driving — Precision: 0.88, Recall: 0.82, F1: 0.85
- Signal Jumping — Precision: 0.85, Recall: 0.79, F1: 0.82
- Mobile Phone Usage — Precision: 0.83, Recall: 0.76, F1: 0.79
- Triple Riding — Precision: 0.90, Recall: 0.85, F1: 0.87

## How it works (high level)
1. Video frames are sampled and passed through a YOLOv8 detector.
2. Detected boxes are tracked (track IDs maintained) and cropped for specialized classifiers (helmet/phone) or pose analysis.
3. Heuristic rules or secondary models determine violations and junction events.
4. Results are written to `output.json` and consumed by the dashboard for visualization.

## For Final Submission (what to replace)
This starter pipeline uses placeholder heuristics for some violations. For a production-quality submission, replace heuristics with trained models using official Indian datasets (MoRTH, NCRB, IRAD) and document the training data and weights under `models/`.

## Need help?
- Want me to integrate training scripts, add model weights, or capture dashboard screenshots for the report? Say the word and I'll do the next step.

---
Generated README — updated to be clear, friendly, and submission-ready.
