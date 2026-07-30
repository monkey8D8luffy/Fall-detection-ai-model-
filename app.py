import streamlit as st
import numpy as np
import joblib
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile
import os
from collections import deque
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI-Powered Elderly Fall Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# LIQUID GLASS THEME — YELLOW & WHITE
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #FFFDF6 0%, #FFF8E1 45%, #FFFFFF 100%);
    background-attachment: fixed;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ---------- Glass header banner ---------- */
.glass-header {
    position: relative;
    background: linear-gradient(135deg, rgba(255,255,255,0.65), rgba(255,236,179,0.55));
    backdrop-filter: blur(24px) saturate(200%);
    -webkit-backdrop-filter: blur(24px) saturate(200%);
    border: 1px solid rgba(255,193,7,0.4);
    border-radius: 28px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 10px 40px rgba(255,193,7,0.18), inset 0 1px 1px rgba(255,255,255,0.7);
    overflow: hidden;
}
.glass-header-content {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    position: relative;
    z-index: 2;
}
.glass-header-icon {
    font-size: 2.8rem;
    filter: drop-shadow(0 4px 10px rgba(255,160,0,0.4));
}
.glass-header-title {
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(135deg, #B8860B, #FFA000);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.glass-header-subtitle {
    font-size: 0.95rem;
    color: #8a6d00;
    font-weight: 400;
    margin-top: 2px;
}
.glass-header-glow {
    position: absolute;
    top: -60%;
    right: -8%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(255,193,7,0.35), transparent 70%);
    z-index: 1;
    pointer-events: none;
}

/* ---------- Liquid glass tab bar (top navigation) ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255,255,255,0.45);
    backdrop-filter: blur(24px) saturate(200%);
    -webkit-backdrop-filter: blur(24px) saturate(200%);
    border-radius: 100px;
    padding: 8px;
    border: 1px solid rgba(255,193,7,0.4);
    box-shadow: 0 8px 30px rgba(255,193,7,0.15);
    width: fit-content;
    margin: 0 auto 1.6rem auto;
}
.stTabs [data-baseweb="tab"] {
    height: 46px;
    border-radius: 100px !important;
    background-color: transparent;
    color: #8a6d00;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0 22px;
    transition: all 0.25s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255,193,7,0.15);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #FFCA28, #FFA000) !important;
    color: #402d00 !important;
    box-shadow: 0 4px 18px rgba(255,160,0,0.4);
}
.stTabs [data-baseweb="tab-highlight"] { background-color: transparent; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- Glass cards (any st.container(key=...)) ---------- */
div[class*="st-key-"] {
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid rgba(255,193,7,0.35);
    border-radius: 22px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(255,193,7,0.12), inset 0 1px 1px rgba(255,255,255,0.6);
    transition: all 0.3s ease;
}
div[class*="st-key-"]:hover {
    box-shadow: 0 12px 40px rgba(255,193,7,0.18), inset 0 1px 1px rgba(255,255,255,0.7);
}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.5);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 18px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(255,193,7,0.3);
    box-shadow: 0 4px 20px rgba(255,193,7,0.1);
}
[data-testid="stMetricValue"] { color: #7a5c00; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #a3843a; }

/* ---------- Buttons ---------- */
.stButton>button {
    background: linear-gradient(135deg, #FFC107 0%, #FFA000 100%);
    color: #402d00;
    font-weight: 600;
    border: none;
    border-radius: 14px;
    padding: 0.6rem 1.6rem;
    box-shadow: 0 4px 15px rgba(255,160,0,0.35);
    transition: all 0.25s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(255,160,0,0.45);
    color: #402d00;
}

/* ---------- Radio pills ---------- */
div[role="radiogroup"] {
    background: rgba(255,255,255,0.5);
    backdrop-filter: blur(14px);
    border-radius: 100px;
    padding: 6px 10px;
    border: 1px solid rgba(255,193,7,0.3);
}
div[role="radiogroup"] label {
    margin: 2px 4px;
}

/* ---------- Alerts ---------- */
.stAlert {
    border-radius: 16px !important;
    backdrop-filter: blur(12px);
}

/* ---------- Progress bar ---------- */
.stProgress > div > div > div > div {
    background-image: linear-gradient(90deg, #FFC107, #FFECB3);
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.5);
    border-radius: 16px;
    border: 1.5px dashed rgba(255,193,7,0.5);
}

/* ---------- Small helper boxes ---------- */
.tip-box {
    background: rgba(255, 236, 179, 0.4);
    border-left: 3px solid #FFC107;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #6b5300;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# ASSET LOADING
# ============================================================
@st.cache_resource
def load_assets():
    yolo = YOLO('yolov8n-pose.pt')
    model_path = 'elderly_fall_20frame_compressed.pkl'

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = None
    return yolo, model

yolo_model, rf_model = load_assets()
model_loaded = rf_model is not None

# ============================================================
# SESSION STATE
# ============================================================
for key, default in [('total_scans', 0), ('fall_count', 0), ('normal_count', 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="glass-header">
    <div class="glass-header-glow"></div>
    <div class="glass-header-content">
        <div class="glass-header-icon">🛡️</div>
        <div>
            <div class="glass-header-title">AI-Powered Elderly Fall Detection</div>
            <div class="glass-header-subtitle">Real-Time Pose Intelligence for Healthcare Monitoring</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TOP LIQUID GLASS TAB BAR
# ============================================================
tab_live, tab_perf, tab_about = st.tabs(
    ["🎥  Live Detection", "📊  Model Performance", "🛡️  Project Overview"]
)

# ============================================================
# TAB 1 — LIVE DETECTION DASHBOARD
# ============================================================
with tab_live:
    if not model_loaded:
        st.error(
            "⚠️ Model artifact could not be loaded. Please ensure "
            "`elderly_fall_20frame_compressed.pkl` is present in the project directory."
        )

    with st.container(key="live_metrics_row"):
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Scans Processed", st.session_state['total_scans'])
        m2.metric("Emergency Falls Detected", st.session_state['fall_count'])
        m3.metric("Normal States Recorded", st.session_state['normal_count'])

    left, right = st.columns([2, 1])

    with left:
        with st.container(key="live_input_card"):
            st.markdown("#### 🔍 Choose Input Source")
            upload_type = st.radio(
                "Select Input Type",
                ["🖼️ Image Frame", "🎞️ Video Clip", "📷 Live Webcam"],
                horizontal=True,
                label_visibility="collapsed"
            )

        # ---------------- IMAGE UPLOAD ----------------
        if upload_type == "🖼️ Image Frame":
            with st.container(key="live_image_card"):
                uploaded_file = st.file_uploader(
                    "Upload patient monitoring frame", type=["jpg", "jpeg", "png"]
                )

                if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.image(image, caption="Uploaded Patient Frame", use_container_width=True)

                    if model_loaded and st.button("✨ Run AI Pose & Activity Analysis"):
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

                                    if prediction == 'fall':
                                        st.session_state['fall_count'] += 1
                                        st.error(
                                            "🚨 **EMERGENCY ALERT: FALL DETECTED!** "
                                            "Immediate caregiver intervention required."
                                        )
                                    else:
                                        st.session_state['normal_count'] += 1
                                        st.success(f"✅ Status Stable — Classified Activity: **{prediction.upper()}**")

                                    st.markdown("##### Model Confidence Breakdown")
                                    colors = ['#E53935' if c == 'fall' else '#FFC107' for c in rf_model.classes_]
                                    fig = go.Figure(go.Bar(
                                        x=[p * 100 for p in probabilities],
                                        y=[c.upper() for c in rf_model.classes_],
                                        orientation='h',
                                        marker=dict(color=colors),
                                        text=[f"{p*100:.1f}%" for p in probabilities],
                                        textposition='outside'
                                    ))
                                    fig.update_layout(
                                        xaxis=dict(range=[0, 105], showgrid=False, title="Confidence (%)"),
                                        yaxis=dict(showgrid=False, autorange="reversed"),
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        height=280,
                                        margin=dict(l=10, r=30, t=10, b=30),
                                        font=dict(family="Poppins", color="#5c4400")
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                    annotated_frame = results[0].plot()
                                    st.image(
                                        annotated_frame,
                                        caption="YOLOv8 Pose Estimation Keypoint Overlay",
                                        use_container_width=True
                                    )
                                else:
                                    st.warning("No human skeleton detected in frame. Please upload a clearer view.")
                            else:
                                st.warning("Pose estimation failed to locate anatomical landmarks.")

        # ---------------- VIDEO UPLOAD ----------------
        elif upload_type == "🎞️ Video Clip":
            with st.container(key="live_video_card"):
                uploaded_video = st.file_uploader(
                    "Upload patient monitoring video sequence", type=["mp4", "avi", "mov"]
                )

                if uploaded_video is not None:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(uploaded_video.read())
                    video_path = tfile.name

                    st.video(video_path)

                    if model_loaded and st.button("▶️ Process Video Stream"):
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
                                        st.success(f"✅ Video Frame Status: **{prediction.upper()}**")

                                    st.image(
                                        results[0].plot(),
                                        caption="Processed Video Frame Keypoints",
                                        use_container_width=True
                                    )
                                else:
                                    st.warning("No human skeleton detected in the sampled frame.")
                            else:
                                st.warning("Pose estimation failed on the sampled frame.")
                        else:
                            st.error("Could not read video file stream.")
                        vidcap.release()

        # ---------------- LIVE WEBCAM ----------------
        elif upload_type == "📷 Live Webcam":
            with st.container(key="live_webcam_card"):
                st.warning(
                    "⚠️ **Note:** Live webcam feed requires running this app locally via your "
                    "terminal (`streamlit run app.py`). It will not access your camera while "
                    "hosted on Streamlit Cloud."
                )

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

                                    annotated_frame = results[0].plot()
                                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                                    if len(window) == 20:
                                        motion_vector = np.concatenate(window).reshape(1, -1)
                                        prediction = rf_model.predict(motion_vector)[0]

                                        if prediction == 'fall':
                                            cv2.putText(
                                                annotated_frame, "ALERT: FALL DETECTED", (20, 50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3
                                            )
                                        else:
                                            cv2.putText(
                                                annotated_frame, f"Status: {prediction.upper()}", (20, 50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (7, 193, 255), 3
                                            )

                                    FRAME_WINDOW.image(annotated_frame)
                                else:
                                    FRAME_WINDOW.image(frame_rgb)
                            else:
                                FRAME_WINDOW.image(frame_rgb)

                        cap.release()

    with right:
        with st.container(key="live_insights_card"):
            st.markdown("#### 📈 Session Insights")
            if st.session_state['total_scans'] > 0:
                fig_donut = go.Figure(go.Pie(
                    labels=['Falls Detected', 'Normal Activity'],
                    values=[st.session_state['fall_count'], st.session_state['normal_count']],
                    hole=0.62,
                    marker=dict(colors=['#E53935', '#FFC107'], line=dict(color='#FFFFFF', width=2)),
                    textinfo='percent',
                    textfont=dict(color='#402d00')
                ))
                fig_donut.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=260,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=True,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=-0.25,
                        font=dict(family="Poppins", size=11, color="#5c4400")
                    ),
                    font=dict(family="Poppins")
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.info("Run a scan to see live session analytics here.")

            st.markdown("---")
            st.markdown("""
            <div class="tip-box">
            💡 <b>Tip:</b> For best pose detection accuracy, ensure the full body is visible
            and well-lit in the frame.
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 2 — MODEL PERFORMANCE & METRICS
# ============================================================
with tab_perf:
    st.markdown("### 📊 Model Evaluation & Validation Metrics")
    st.write(
        "Evaluating the system using validation/test split data to verify precision, "
        "recall, and overall accuracy."
    )

    with st.container(key="perf_metrics_row"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Accuracy", "94.44%")
        c2.metric("Fall Precision", "94%")
        c3.metric("Fall Recall", "97%")
        c4.metric("F1-Score (Fall)", "0.96")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        with st.container(key="perf_gauge_card"):
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=94.44,
                number={'suffix': "%", 'font': {'size': 36, 'color': '#7a5c00'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#c9a227'},
                    'bar': {'color': '#FFC107'},
                    'bgcolor': 'rgba(255,255,255,0.2)',
                    'borderwidth': 1,
                    'bordercolor': 'rgba(255,193,7,0.4)',
                    'steps': [
                        {'range': [0, 60], 'color': 'rgba(255,235,180,0.5)'},
                        {'range': [60, 85], 'color': 'rgba(255,213,79,0.5)'},
                        {'range': [85, 100], 'color': 'rgba(255,193,7,0.6)'}
                    ],
                },
                title={'text': "Overall Accuracy", 'font': {'size': 16, 'color': '#5c4400'}}
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                height=300,
                margin=dict(l=20, r=20, t=50, b=10),
                font=dict(family="Poppins")
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

    with chart_col2:
        with st.container(key="perf_bar_card"):
            fig_bar = go.Figure(go.Bar(
                x=['Precision', 'Recall', 'F1-Score'],
                y=[94, 97, 96],
                marker=dict(color=['#FFD54F', '#FFC107', '#FFA000']),
                text=['94%', '97%', '0.96'],
                textposition='outside',
                width=0.5
            ))
            fig_bar.update_layout(
                title=dict(text="Fall-Class Detection Metrics", font=dict(size=16, color="#5c4400")),
                yaxis=dict(range=[0, 110], title="Score", showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                font=dict(family="Poppins", color="#5c4400")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        with st.container(key="perf_confusion_card"):
            st.markdown("##### Confusion Matrix Analysis")
            if os.path.exists('confusion_matrix_final.png'):
                st.image(
                    'confusion_matrix_final.png',
                    caption="Final 20-Frame Temporal Confusion Matrix",
                    use_container_width=True
                )
            else:
                st.warning(
                    "Confusion matrix image not found. Add `confusion_matrix_final.png` "
                    "to your project folder to display it here."
                )

    with img_col2:
        with st.container(key="perf_distribution_card"):
            st.markdown("##### Dataset Class Distribution")
            if os.path.exists('activity_class_distribution.png'):
                st.image(
                    'activity_class_distribution.png',
                    caption="Balanced Dataset Sequence Distribution",
                    use_container_width=True
                )
            else:
                st.warning(
                    "Class distribution chart not found. Add `activity_class_distribution.png` "
                    "to your project folder to display it here."
                )

# ============================================================
# TAB 3 — PROJECT OVERVIEW & MAINTENANCE
# ============================================================
with tab_about:
    with st.container(key="about_hero_card"):
        st.markdown("## 🛡️ AI-Powered Elderly Fall Detection System")
        st.markdown("#### Formative Assessment-2: Implementation & Deployment")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(key="about_profile_card"):
            st.markdown("""
            **Student Profile**
            - **Name:** Saurav Kamble
            - **Program:** IB Career-related Programme (IBCP)
            - **Specialization:** Artificial Intelligence Pathway

            **Intended Learning Outcomes**
            - Implement computer vision and pose estimation to monitor patient safety.
            - Build and deploy real-time healthcare dashboards using Streamlit.
            - Train multi-class classifiers for human activity recognition.
            """)
    with col2:
        with st.container(key="about_architecture_card"):
            st.markdown("""
            **System Architecture**
            1. **Pose Estimation** — YOLOv8 Pose extracts 17 anatomical keypoints.
            2. **Temporal Window** — Gathers sequences over time to evaluate trajectory.
            3. **Classification** — Random Forest model identifies 5 activity classes
               (`fall`, `walking`, `sitting`, `standing`, `normal`).
            4. **Alert System** — Automatically generates emergency notifications when a
               fall is detected.
            """)

    with st.container(key="about_maintenance_card"):
        st.markdown("#### 🔧 System Monitoring & Future Maintenance")
        st.write("""
        To maintain high reliability in real-world deployment, hospitals and caregivers
        should consider:
        - **Periodic Retraining** — updating the model with new patient activity annotations.
        - **CCTV Integration** — expanding the pipeline to support continuous live multi-camera feeds.
        - **False Alert Mitigation** — refining temporal confidence thresholds to minimize
          false emergency notifications.
        """)
