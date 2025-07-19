import streamlit as st
import numpy as np
import tempfile
import os
from PIL import Image
import base64
import io
import time

# Try to import OpenCV with error handling
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    st.error("""
    ⚠️ OpenCV (cv2) is not available. This might be due to deployment environment limitations.
    
    **Solutions:**
    1. Try refreshing the page
    2. Check if the app is still deploying
    3. Contact support if the issue persists
    
    The app will work with limited functionality without OpenCV.
    """)
    OPENCV_AVAILABLE = False

# Try to import YOLO with error handling
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    st.error("""
    ⚠️ YOLO model is not available. This might be due to deployment environment limitations.
    
    **Solutions:**
    1. Try refreshing the page
    2. Check if the app is still deploying
    3. Contact support if the issue persists
    """)
    YOLO_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🌊 Underwater Trash Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ocean theme
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #006994 0%, #003366 50%, #001a33 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #006994 0%, #003366 50%, #001a33 100%);
    }
    .stButton > button {
        background: linear-gradient(45deg, #006994, #87ceeb);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #005a7a, #5f9ea0);
        transform: translateY(-2px);
    }
    .uploadedFile {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
    }
    .detection-box {
        border: 2px solid #00ff00;
        background: rgba(0, 255, 0, 0.1);
        color: #00ff00;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
    }
    h1, h2, h3 {
        color: #87ceeb !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .stMarkdown {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    """Load the YOLO model"""
    if not YOLO_AVAILABLE:
        st.error("YOLO is not available. Cannot load model.")
        return None
    
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = load_model()

if 'processed_frames' not in st.session_state:
    st.session_state.processed_frames = []

# Main title
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1>🌊 Underwater Trash Detection System</h1>
    <p style="font-size: 1.2rem; opacity: 0.9;">Protecting our oceans, one detection at a time</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Detection Settings")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Minimum confidence score for detections"
    )
    
    frame_skip = st.slider(
        "Frame Skip Interval",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        help="Process every Nth frame (higher = faster processing)"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    if st.session_state.processed_frames:
        total_detections = sum(frame['detections'] for frame in st.session_state.processed_frames)
        st.metric("Total Frames Processed", len(st.session_state.processed_frames))
        st.metric("Total Detections", total_detections)
        st.metric("Avg Detections/Frame", f"{total_detections/len(st.session_state.processed_frames):.1f}")

# Main content
tab1, tab2, tab3 = st.tabs(["📹 Video Upload", "📷 Live Detection", "📊 Results"])

with tab1:
    st.markdown("### 📹 Upload Underwater Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload an underwater video to detect trash"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name
        
        if st.button("🔍 Process Video", type="primary"):
            if not OPENCV_AVAILABLE:
                st.error("OpenCV is not available. Video processing requires OpenCV.")
            elif st.session_state.model is None:
                st.error("Model not loaded. Please check your model file.")
            else:
                with st.spinner("Processing video frames..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process video
                    cap = cv2.VideoCapture(video_path)
                    frame_count = 0
                    processed_count = 0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    st.session_state.processed_frames = []
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        
                        # Process every Nth frame
                        if frame_count % frame_skip == 0:
                            # Run inference
                            results = st.session_state.model(frame, conf=confidence_threshold)
                            
                            # Process results
                            detections = 0
                            for result in results:
                                boxes = result.boxes
                                if boxes is not None:
                                    for box in boxes:
                                        # Get box coordinates
                                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                        confidence = box.conf[0].cpu().numpy()
                                        
                                        # Draw bounding box
                                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                        
                                        # Add label
                                        label = f"Trash: {confidence:.2f}"
                                        cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                        
                                        detections += 1
                            
                            # Convert frame to PIL Image
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(frame_rgb)
                            
                            # Store frame data
                            st.session_state.processed_frames.append({
                                'frame_number': frame_count,
                                'image': pil_image,
                                'detections': detections,
                                'confidence': confidence_threshold
                            })
                            
                            processed_count += 1
                            
                            # Update progress
                            progress = min(frame_count / total_frames, 1.0)
                            progress_bar.progress(progress)
                            status_text.text(f"Processed {processed_count} frames...")
                    
                    cap.release()
                    
                    # Clean up temporary file
                    os.unlink(video_path)
                    
                    st.success(f"✅ Video processing complete! Processed {processed_count} frames.")
                    
                    # Show summary
                    if st.session_state.processed_frames:
                        total_detections = sum(frame['detections'] for frame in st.session_state.processed_frames)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Frames Processed", processed_count)
                        with col2:
                            st.metric("Total Detections", total_detections)
                        with col3:
                            st.metric("Avg Detections/Frame", f"{total_detections/processed_count:.1f}")

with tab2:
    st.markdown("### 📷 Live Webcam Detection")
    
    # Camera input
    camera_input = st.camera_input("Take a photo for detection")
    
    if camera_input is not None:
        # Convert to PIL Image
        image = Image.open(camera_input)
        
        if st.button("🔍 Detect Trash", type="primary"):
            if not OPENCV_AVAILABLE:
                st.error("OpenCV is not available. Image processing requires OpenCV.")
            elif st.session_state.model is None:
                st.error("Model not loaded. Please check your model file.")
            else:
                with st.spinner("Detecting trash..."):
                    # Convert PIL to OpenCV format
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Run inference
                    results = st.session_state.model(image_cv, conf=confidence_threshold)
                    
                    # Process results
                    detections = 0
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                # Get box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                confidence = box.conf[0].cpu().numpy()
                                
                                # Draw bounding box
                                cv2.rectangle(image_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                
                                # Add label
                                label = f"Trash: {confidence:.2f}"
                                cv2.putText(image_cv, label, (int(x1), int(y1) - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                detections += 1
                    
                    # Convert back to PIL for display
                    result_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                    result_pil = Image.fromarray(result_image)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Original Image", use_column_width=True)
                    with col2:
                        st.image(result_pil, caption=f"Detection Results ({detections} items found)", use_column_width=True)
                    
                    if detections > 0:
                        st.success(f"🎯 Found {detections} trash items!")
                    else:
                        st.info("✅ No trash detected in this image.")

with tab3:
    st.markdown("### 📊 Detection Results")
    
    if not st.session_state.processed_frames:
        st.info("No processed frames yet. Upload a video or use live detection to see results.")
    else:
        st.markdown(f"#### 📈 Analysis Summary")
        
        # Statistics
        total_frames = len(st.session_state.processed_frames)
        total_detections = sum(frame['detections'] for frame in st.session_state.processed_frames)
        avg_detections = total_detections / total_frames if total_frames > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Frames", total_frames)
        with col2:
            st.metric("Total Detections", total_detections)
        with col3:
            st.metric("Avg Detections/Frame", f"{avg_detections:.1f}")
        with col4:
            st.metric("Detection Rate", f"{(total_detections/total_frames)*100:.1f}%" if total_frames > 0 else "0%")
        
        st.markdown("#### 🖼️ Processed Frames")
        
        # Display frames in a grid
        frames_per_row = 3
        for i in range(0, len(st.session_state.processed_frames), frames_per_row):
            cols = st.columns(frames_per_row)
            for j in range(frames_per_row):
                if i + j < len(st.session_state.processed_frames):
                    frame = st.session_state.processed_frames[i + j]
                    with cols[j]:
                        st.image(
                            frame['image'],
                            caption=f"Frame {frame['frame_number']} - {frame['detections']} detections",
                            use_column_width=True
                        )
        
        # Clear results button
        if st.button("🗑️ Clear Results"):
            st.session_state.processed_frames = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; opacity: 0.7;">
    <p>🌊 Underwater Trash Detection System | Built with Streamlit & YOLO</p>
    <p>Protecting our oceans, one detection at a time ♻️</p>
</div>
""", unsafe_allow_html=True) 