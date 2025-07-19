from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64
from PIL import Image
import io
import json
from ultralytics import YOLO
import tempfile
import uuid
from trash_classes import get_class_name_short, get_class_color, update_mapping_from_model

app = Flask(__name__)
CORS(app)

# Load the trained model
model = YOLO('best.pt')

# Debug: Print model class names if available and update mapping
try:
    if hasattr(model, 'names'):
        print("🔍 Model class names:", model.names)
        # Update our mapping to match the model's actual class names
        update_mapping_from_model(model.names)
    else:
        print("⚠️ No class names found in model")
except Exception as e:
    print(f"❌ Error accessing model names: {e}")

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store processed frames for video recreation
processed_frames_storage = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'Underwater Trash Detection System is running'})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        # Save the uploaded video
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.mp4")
        video_file.save(video_path)
        
        # Process the video
        results = process_video(video_path)
        
        # Clean up the uploaded file
        os.remove(video_path)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_video(video_path):
    """Process video frame by frame and detect trash"""
    cap = cv2.VideoCapture(video_path)
    frame_results = []
    frame_count = 0
    processed_frames = []  # Store frames for video recreation
    
    # Get original video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 5th frame to avoid too many results
        if frame_count % 5 == 0:
            # Create a copy of the frame for processing
            processed_frame = frame.copy()
            
            # Run inference on the frame
            results = model(frame)
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Debug: Print class ID being detected
                        print(f"🔍 Detected Class ID: {class_id}")
                        
                        # Get class name and color
                        class_name = get_class_name_short(class_id)
                        color = get_class_color(class_id)
                        
                        # Draw bounding box on processed frame
                        cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Add label
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(processed_frame, label, (int(x1), int(y1) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Convert frame to base64 for frontend
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            frame_results.append({
                'frame_number': frame_count,
                'image': frame_base64,
                'detections': len(results[0].boxes) if results[0].boxes is not None else 0
            })
            
            # Store processed frame for video recreation
            processed_frames.append(processed_frame)
        
        frame_count += 1
    
    cap.release()
    
    # Generate unique session ID for this processing
    session_id = str(uuid.uuid4())
    
    # Store processed frames and video properties
    processed_frames_storage[session_id] = {
        'frames': processed_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': frame_count,
        'processed_frames': len(processed_frames)
    }
    
    return {
        'frames': frame_results,
        'total_frames': frame_count,
        'processed_frames': len(frame_results),
        'session_id': session_id
    }

@app.route('/recreate_video', methods=['POST'])
def recreate_video():
    """Recreate video from processed frames"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id not in processed_frames_storage:
            return jsonify({'error': 'Invalid session ID or no processed frames found'}), 400
        
        storage_data = processed_frames_storage[session_id]
        frames = storage_data['frames']
        fps = storage_data['fps']
        width = storage_data['width']
        height = storage_data['height']
        
        # Create temporary video file
        video_filename = f"recreated_video_{session_id}.mp4"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Write frames to video
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        # Return video file
        return send_file(video_path, as_attachment=True, download_name=f"detected_trash_video.mp4")
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a single frame from webcam"""
    try:
        data = request.get_json()
        frame_data = data['frame']
        
        # Remove data URL prefix
        frame_data = frame_data.split(',')[1]
        
        # Decode base64 image
        image_data = base64.b64decode(frame_data)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = model(frame)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': get_class_name_short(class_id)
                    })
        
        return jsonify({'detections': detections})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port) 