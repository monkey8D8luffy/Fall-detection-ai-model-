# 🛡️ AI-Powered Elderly Fall Detection System (FA-2)

## Project Overview
Developed as part of the International Baccalaureate Career-related Programme (IBCP) Artificial Intelligence coursework, this project builds and deploys an end-to-end computer vision and machine learning healthcare monitoring dashboard[span_11](start_span)[span_11](end_span). The system detects elderly postures, tracks movement trajectories using temporal windows, classifies activities (`fall`, `walking`, `sitting`, `standing`, `normal`), and triggers instant emergency alerts[span_12](start_span)[span_12](end_span).

## Features Included
* **Real-Time Pose Estimation:** Powered by YOLOv8 Pose to map anatomical body keypoints[span_13](start_span)[span_13](end_span).
* **Temporal Sequence Analysis:** Evaluates movement over multi-frame sliding windows (20-frame configuration) to accurately distinguish dynamic activities like walking from static postures.
* **Emergency Alert System:** Automatically displays high-priority warning notifications when a fall incident is classified[span_14](start_span)[span_14](end_span).
* **Interactive Streamlit Dashboard:** Supports image and video file uploads, visualizes tracking analytics, and displays prediction confidence scores[span_15](start_span)[span_15](end_span).
* **Rigorous Evaluation Metrics:** Integrates overall accuracy, precision, recall, F1-score tables, confusion matrices, and distribution charts[span_16](start_span)[span_16](end_span).

## Repository Structure
* `app.py`: Main Streamlit cloud web application code.
* `requirements.txt`: Python package dependencies.
* `elderly_fall_20frame_model.pkl`: Trained Random Forest temporal classification model artifact.
* `confusion_matrix_final.png` & `activity_class_distribution.png`: Evaluation charts for model analysis.

## Local Installation & Execution
1. Clone the repository or download project files.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
