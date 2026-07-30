import streamlit as st
import numpy as np
import joblib
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile
import os
from collections import deque

# Page Configuration
st.set_page_config(
    page_title="AI-Powered Elderly Fall Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling & CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# Load Model and YOLO Assets
@st.cache_resource
def load_assets():
    yolo = YOLO('yolov8n-pose.pt')
    # Updated to load your new compressed model file
    model_path = 'elderly_fall_20frame_compressed.pkl'
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        # Fallback just in case
        model = None 
    return yolo, model

yolo_model, rf_model = load_assets()
model_loaded = rf_model is not None

# Sidebar Navigation
st.sidebar.title("🏥 Healthcare Monitoring")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "Navigation Menu", 
    ["Live Detection Dashboard", "Model Performance & Metrics", "Project Overview & Maintenance"]
)

# Session State for Analytics Counters
if 'total_scans' not in st.session_state:
    st.session_state['total_scans'] = 0
if 'fall_count' not in st.session_state:
    st.session_state['fall_count'] = 0
if 'normal_count' not in st.session_state:
    st.session_state['normal_count'] = 0

# ==========================================
# PAGE 1: PROJECT OVERVIEW & MAINTENANCE
# ==========================================
if app_mode == "Project Overview & Maintenance":
    st.title("🛡️ AI-Powered Elderly Fall Detection System")
    st.markdown("### Formative Assessment-2: Implementation & Deployment")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Student Profile:**
        * **Name:** Saurav Kamble
        * **Program:** IB Career-related Programme (IBCP)
        * **Specialization:** Artificial Intelligence Pathway
        
        **Intended Learning Outcomes:**
        * Implement computer vision and pose estimation to monitor patient safety.
        * Build and deploy real-time healthcare dashboards using Streamlit.
        * Train multi-class classifiers for human activity recognition.
        """)
    with col2:
        st.info("""
        **System Architecture:**
        1. **Pose Estimation:** YOLOv8 Pose extracts 17 anatomical keypoints.
        2. **Temporal Window:** Gathers sequences over time to evaluate trajectory.
        3. **Classification:** Random Forest Model identifies 5 distinct classes (`fall`, `walking`, `sitting`, `standing`, `normal`).
        4. **Alert System:** Automatically generates emergency notifications when a fall is detected.
        """)
        
    st.markdown("---")
    st.subheader("🔧 System Monitoring & Future Maintenance")
    st.write("""
    To maintain high reliability in real-world deployment, hospitals and caregivers should consider:
    * **Periodic Retraining:** Updating the model with new patient activity annotations.
    * **CCTV Integration:** Expanding the pipeline to support continuous live multi-camera feeds.
    * **False Alert Mitigation:** Refining temporal confidence thresholds to minimize false emergency notifications.
    """)

# ==========================================
# PAGE 2: MODEL PERFORMANCE & METRICS
# ==========================================
elif app_mode == "Model Performance & Metrics":
    st.title("📊 Model Evaluation & Validation Metrics")
    st.write("Evaluating the system using validation/test split data to verify precision, recall, and overall accuracy.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Accuracy", "94.44%")
    col2.metric("Fall Precision", "94%")
    col3.metric("Fall Recall", "97%")
    col4.metric("F1-Score (Fall)", "0.96")
    
    st.markdown("---")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Confusion Matrix Analysis")
        if os.path.exists('confusion_matrix_final.png'):
            st.image('confusion_matrix_final.png', caption="Final 20-Frame Temporal Confusion Matrix", use_column_width=True)
        else:
            st.warning("Confusion matrix image file not found. Upload it to your repository to display it here.")
            
    with col_b:
        st.subheader("Dataset Class Distribution")
        if os.path.exists('activity_class_distribution.png'):
            st.image('activity_class_distribution.png', caption="Balanced Dataset Sequence Distribution", use_column_width=True)
        else:
            st.warning("Class distribution chart image not found.")

# ==========================================
# PAGE 3: LIVE DETECTION DASHBOARD
# ==========================================
elif app_mode == "Live Detection Dashboard":
    st.title("🎥 Real-Time Healthcare Monitoring Dashboard")
    st.write("Upload an image, process a video, or activate your local webcam to evaluate posture and classify activity.")
    
    if not model_loaded:
        st.error("Model artifact could not be loaded. Please ensure `elderly_fall_20frame_compressed.pkl` is uploaded to your repository.")
    
    # Analytics Overview Metrics Bar
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Scans Processed", st.session_state['total_scans'])
    m2.metric("Emergency Falls Detected", st.session_state['fall_count'], delta_color="inverse")
    m3.metric("Normal States Recorded", st.session_state['normal_count'])
    
    st.markdown("---")
    
    upload_type = st.radio("Select Input Type", ["Image Frame Upload", "Video Clip Upload", "Live Webcam (Local Only)"])
    
    # --- IMAGE UPLOAD LOGIC ---
    if upload_type == "Image Frame Upload":
        uploaded_file = st.file_uploader("Upload patient monitoring frame...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None and model_loaded:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Patient Frame", use_column_width=True)
            
            if st.button("Run AI Pose & Activity Analysis"):
                st.session_state['total_scans'] += 1
                with st.spinner("Extracting keypoints and running temporal classification..."):
                    results = yolo_model(image, verbose=False)
                    
                    if len(results) > 0 and results[0].keypoints is not None:
                        kpts = results[0].keypoints.data
                        if len(kpts) > 0:
                            features = kpts[0].cpu().numpy().flatten()
                            features_input = np.tile(features, (20, 1)).flatten().reshape(1, -1)
                            
                            prediction = rf_model.predict(features_input)[0]
                            probabilities = rf_model.predict_proba(features_input)[0]
                            
                            st.markdown("---")
                            st.subheader("Analysis Results:")
                            
                            if prediction == 'fall':
                                st.session_state['fall_count'] += 1
                                st.error("🚨 **EMERGENCY ALERT: FALL DETECTED!** Immediate caregiver intervention required!")
                            else:
                                st.session_state['normal_count'] += 1
                                st.success(f"Status Stable. Classified Activity: **{prediction.upper()}**")
                                
                            st.write("### Model Confidence Breakdown:")
                            for cls, prob in zip(rf_model.classes_, probabilities):
                                st.progress(float(prob), text=f"{cls.upper()}: {prob*100:.1f}%")
                                
                            annotated_frame = results[0].plot()
                            st.image(annotated_frame, caption="YOLOv8 Pose Estimation Keypoint Overlay", use_column_width=True)
                        else:
                            st.warning("No human skeleton detected in frame. Please upload a clear view.")
                    else:
                        st.warning("Pose estimation failed to locate anatomical landmarks.")

    # --- VIDEO UPLOAD LOGIC ---
    elif upload_type == "Video Clip Upload":
        uploaded_video = st.file_uploader("Upload patient monitoring video sequence...", type=["mp4", "avi", "mov"])
        
        if uploaded_video is not None and model_loaded:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            
            st.video(video_path)
            
            if st.button("Process Video Stream"):
                st.session_state['total_scans'] += 1
                vidcap = cv2.VideoCapture(video_path)
                success, frame = vidcap.read()
                
                if success:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    
                    results = yolo_model(pil_img, verbose=False)
                    if len(results) > 0 and results[0].keypoints is not None:
                        kpts = results[0].keypoints.data
                        if len(kpts) > 0:
                            features = kpts[0].cpu().numpy().flatten()
                            features_input = np.tile(features, (20, 1)).flatten().reshape(1, -1)
                            
                            prediction = rf_model.predict(features_input)[0]
                            
                            if prediction == 'fall':
                                st.session_state['fall_count'] += 1
                                st.error("🚨 **EMERGENCY ALERT: FALL DETECTED IN VIDEO STREAM!**")
                            else:
                                st.session_state['normal_count'] += 1
                                st.success(f"Video Frame Status: **{prediction.upper()}**")
                                
                            st.image(results[0].plot(), caption="Processed Video Frame Keypoints")
                else:
                    st.error("Could not read video file stream.")

    # --- LIVE WEBCAM LOGIC ---
    elif upload_type == "Live Webcam (Local Only)":
        st.warning("⚠️ **Note:** Live webcam feed requires running this app locally via your terminal (`streamlit run app.py`). It will not access your camera while hosted on the Streamlit Cloud.")
        
        if model_loaded:
            run_camera = st.checkbox("Turn On Webcam")
            FRAME_WINDOW = st.image([])
            
            if run_camera:
                cap = cv2.VideoCapture(0)
                window = deque(maxlen=20)
                
                while run_camera:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to access webcam.")
                        break
                        
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    
                    results = yolo_model(pil_img, verbose=False)
                    
                    if len(results) > 0 and results[0].keypoints is not None:
                        kpts = results[0].keypoints.data
                        if len(kpts) > 0:
                            features = kpts[0].cpu().numpy().flatten()
                            window.append(features)
                            
                            # Draw YOLO skeleton on the live feed
                            annotated_frame = results[0].plot()
                            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            
                            if len(window) == 20:
                                motion_vector = np.concatenate(window).reshape(1, -1)
                                prediction = rf_model.predict(motion_vector)[0]
                                
                                # Overlay text on the video feed
                                if prediction == 'fall':
                                    cv2.putText(annotated_frame, "ALERT: FALL DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                                else:
                                    cv2.putText(annotated_frame, f"Status: {prediction.upper()}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                    
                            FRAME_WINDOW.image(annotated_frame)
                        else:
                            FRAME_WINDOW.image(frame_rgb)
                    else:
                        FRAME_WINDOW.image(frame_rgb)
                        
                cap.release()
