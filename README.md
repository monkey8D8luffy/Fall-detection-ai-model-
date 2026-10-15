# 🛡️ SafeFall AI — Elderly Fall Detection & Healthcare Monitoring System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://o9rxmeapp9mnskqwqqzfc9.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![YOLOv8 Pose](https://img.shields.io/badge/YOLOv8-Pose%20Estimation-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Scikit-Learn](https://img.shields.io/badge/Model-Random%20Forest-green.svg)](https://scikit-learn.org/)

> **Real-Time Pose Intelligence for Contactless Elderly Safety Monitoring**  
> Developed as part of the **International Baccalaureate Career-related Programme (IBCP) — Artificial Intelligence** coursework at **Ryan Global School, Kharghar**.

🔗 **Live Streamlit Dashboard:** [Launch SafeFall AI Web App](https://o9rxmeapp9mnskqwqqzfc9.streamlit.app/)  
💻 **GitHub Repository:** [monkey8D8luffy/Fall-detection-ai-model-](https://github.com/monkey8D8luffy/Fall-detection-ai-model-)

---

## 📌 Project Overview

Unassisted falls among senior citizens living alone are one of the leading causes of severe trauma and delayed medical emergencies ("long-lie" incidents). Traditional wearable pendants are frequently forgotten, uncharged, or unusable if the wearer is rendered unconscious, while standard CCTV cameras only offer passive recording without active alerting.

**SafeFall AI** is an end-to-end computer vision and machine learning monitoring system that detects falls in real time using standard indoor cameras—requiring zero wearable hardware. By combining **YOLOv8 Pose Estimation** (`17 anatomical keypoints`) with a **Random Forest Classifier** operating on **20-frame temporal sliding windows**, the system tracks biomechanical movement trajectories across five activity states:
* 🚨 **`fall`** *(Emergency Alert Triggered)*
* 🚶 **`walking`**
* 🪑 **`sitting`**
* 🧍 **`standing`**
* ✅ **`normal`** *(General non-fall floor/bending movements)*

---

## ✨ Key Features

* **Real-Time 17-Keypoint Pose Estimation:** Powered by `YOLOv8-Pose` (`yolov8n-pose.pt`) to localize human bounding boxes and map 17 COCO skeletal joints per frame while preserving patient privacy.
* **20-Frame Temporal Window Analysis:** Evaluates continuous movement over 20-frame sliding windows (~0.8 seconds at 25 FPS) to track vertical hip drop velocity, spine inclination angle, and bounding-box aspect ratio inversions—preventing false alarms during controlled sitting or bending.
* **Two-Stage Hybrid Dataset Curation:** Trained and evaluated on **29,261 twenty-frame sequences** extracted from the **Le2i Fall Detection Dataset**, curated using an automated baseline AI pre-classifier followed by a manual human-in-the-loop quality audit.
* **Automated Emergency Alert System:** Immediately renders high-priority visual warning banners and logs session telemetry whenever a fall trajectory is confirmed.
* **3-Tab Interactive Streamlit Dashboard:**
  * **Tab 1 (`Live Detection`):** Supports uploading **Image Frames** (`JPG/PNG`), **Video Clips** (`MP4/AVI/MOV`), or launching a CPU-optimized **Live Webcam** feed with real-time YOLOv8 skeletal overlays.
  * **Tab 2 (`Model Performance`):** Displays validation KPI cards, an interactive accuracy gauge chart, and per-class detection metrics.
  * **Tab 3 (`Project Overview`):** Summarizes student credentials and the 4-stage system architecture.

---

## 🧠 System Architecture

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
