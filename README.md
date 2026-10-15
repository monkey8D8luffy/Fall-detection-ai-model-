# SafeFall AI — Elderly Fall Detection & Healthcare Monitoring System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://o9rxmeapp9mnskqwqqzfc9.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![YOLOv8 Pose](https://img.shields.io/badge/YOLOv8-Pose%20Estimation-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Scikit-Learn](https://img.shields.io/badge/Model-Random%20Forest-green.svg)](https://scikit-learn.org/)

**Real-Time Pose Intelligence for Contactless Elderly Safety Monitoring**  
Developed as part of the **International Baccalaureate Career-related Programme (IBCP) — Artificial Intelligence** coursework at **Ryan Global School, Kharghar**.

**Live Streamlit Dashboard:** [https://o9rxmeapp9mnskqwqqzfc9.streamlit.app/](https://o9rxmeapp9mnskqwqqzfc9.streamlit.app/)  
**GitHub Repository:** [https://github.com/monkey8D8luffy/Fall-detection-ai-model-](https://github.com/monkey8D8luffy/Fall-detection-ai-model-)

---

## Project Overview

Unassisted falls among senior citizens living alone are one of the leading causes of severe trauma and delayed medical emergencies ("long-lie" incidents). Traditional wearable pendants are frequently forgotten, left uncharged, or unusable if the wearer is rendered unconscious, while standard CCTV cameras only offer passive recording without active alerting.

**SafeFall AI** is an end-to-end computer vision and machine learning monitoring system that detects falls in real time using standard indoor cameras—requiring zero wearable hardware. By combining **YOLOv8 Pose Estimation** (`17 anatomical keypoints`) with a **Random Forest Classifier** operating on **20-frame temporal sliding windows**, the system tracks biomechanical movement trajectories across five activity states:
* **`fall`** *(Emergency Alert Triggered)*
* **`walking`**
* **`sitting`**
* **`standing`**
* **`normal`** *(General non-fall floor or bending movements)*

---

## Key Features

* **Real-Time 17-Keypoint Pose Estimation:** Powered by `YOLOv8-Pose` (`yolov8n-pose.pt`) to localize human bounding boxes and map 17 COCO skeletal joints per frame while preserving patient privacy.
* **20-Frame Temporal Window Analysis:** Evaluates continuous movement over 20-frame sliding windows (~0.8 seconds at 25 FPS) to track vertical hip drop velocity, spine inclination angle, and bounding-box aspect ratio inversions—preventing false alarms during controlled sitting or bending.
* **Two-Stage Hybrid Dataset Curation:** Trained and evaluated on **29,261 twenty-frame sequences** extracted from the **Le2i Fall Detection Dataset**, curated using an automated baseline AI pre-classifier followed by a manual human-in-the-loop quality audit.
* **Automated Emergency Alert System:** Immediately renders high-priority visual warning banners and logs session telemetry whenever a fall trajectory is confirmed.
* **3-Tab Interactive Streamlit Dashboard:**
  * **Tab 1 (`Live Detection`):** Supports uploading **Image Frames** (`JPG/PNG`), **Video Clips** (`MP4/AVI/MOV`), or launching a CPU-optimized **Live Webcam** feed with real-time YOLOv8 skeletal overlays.
  * **Tab 2 (`Model Performance`):** Displays validation KPI cards, an interactive accuracy gauge chart, and per-class detection metrics.
  * **Tab 3 (`Project Overview`):** Summarizes student credentials and the 4-stage system architecture.

---

## System Architecture

```text
[Input Video / Image / Live Webcam Stream]
                    │
                    ▼
   [OpenCV 20-Frame Sliding Window Buffer]
   (Resized to 224×224 │ Normalized 0–1)
                    │
                    ▼
      [YOLOv8 Pose Estimation Backbone]
 (Extracts Person Bounding Box + 17 Keypoints)
                    │
                    ▼
   [Biomechanical Trajectory Feature Engine]
 (Hip Velocity + Spine Angle + Aspect Ratio)
                    │
                    ▼
    [Random Forest Temporal Classifier]
   (Predicts: Fall, Walking, Sitting, Standing, Normal)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 [FALL DETECTED]         [NORMAL ACTIVITY]
Triggers Emergency      Updates Safe Session
  Caregiver Alert          Telemetry KPIs
```

---

## Dataset & Model Performance

### 1. Curated Dataset Distribution (`Le2i Fall Dataset` — 20-Frame Windows)
Across 6 indoor surveillance environments (`Coffee_room_01`, `Coffee_room_02`, `Home_01`, `Home_02`, `Lecture_room`, and `Office`), **29,261 sequences** were extracted and partitioned (`70% Train / 15% Validation / 15% Test`):

| Activity Class | Extracted 20-Frame Sequences | Dataset Share (%) |
| :--- | :---: | :---: |
| **`standing`** | 18,352 | 62.72% |
| **`fall`** | 5,390 | 18.42% |
| **`walking`** | 4,364 | 14.91% |
| **`sitting`** | 1,100 | 3.76% |
| **`normal`** | 55 | 0.19% |
| **Total** | **29,261** | **100.00%** |

### 2. Evaluation Benchmark Results
* **Validation Benchmark (Streamlit Dashboard):** `94.44% Overall Accuracy` | `94% Fall Precision` | `97% Fall Recall` | `0.96 Fall F1-Score`
* **Holdout Test Evaluation (`5,763` Temporal Windows):** `94.31% Overall Accuracy` | `98.0% Normalized Fall Recall` (`1,053 / 1,078` falls detected)

| Activity Class | Precision | Recall (Sensitivity) | F1-Score | Test Support (Windows) |
| :--- | :---: | :---: | :---: | :---: |
| **`fall`** | 0.93 | **0.98** | **0.95** | 1,078 |
| **`standing`** | 0.96 | **0.98** | **0.97** | 3,671 |
| **`sitting`** | 0.97 | 0.75 | 0.84 | 220 |
| **`walking`** | 0.88 | 0.81 | 0.84 | 873 |
| **`normal`** | 1.00 | 0.91 | 0.95 | 11 |
| **Weighted Avg** | **0.94** | **0.94** | **0.94** | **5,763** |

---

## Repository Structure

```text
├── app.py                              # Main Streamlit Cloud web application
├── requirements.txt                    # Python package dependencies
├── elderly_fall_20frame_model.pkl      # Trained Random Forest 20-frame classification model
├── yolov8n-pose.pt                     # Pre-trained YOLOv8 Nano Pose estimation weights
├── activity_class_distribution.png     # EDA chart of 29,261 extracted 20-frame sequences
├── confusion_matrix_final.png          # 20-frame temporal confusion matrix (94.31% accuracy)
└── README.md                           # Project documentation
```

---

## Local Installation & Execution

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/monkey8D8luffy/Fall-detection-ai-model-.git](https://github.com/monkey8D8luffy/Fall-detection-ai-model-.git)
   cd Fall-detection-ai-model-
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Application:**
   ```bash
   streamlit run app.py
   ```
   The dashboard will open in your default browser at `http://localhost:8501`.

---

## Author & Academic Credentials

* **Student Name:** Saurav Kamble
* **Candidate Registration Number:** 1000323
* **Program:** International Baccalaureate Career-related Programme (IBCP)
* **Specialization:** Artificial Intelligence Pathway (`Machine Learning & Deep Learning`)
* **School:** Ryan Global School, Kharghar, Navi Mumbai
