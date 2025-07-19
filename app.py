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
from trash_classes import get_class_name_short, get_class_color

app = Flask(__name__)
CORS(app)

# Load the trained model
model = YOLO('best.pt')

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 5th frame to avoid too many results
        if frame_count % 5 == 0:
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
                        
                        # Get class name and color
                        class_name = get_class_name_short(class_id)
                        color = get_class_color(class_id)
                        
                        # Draw bounding box on frame
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Add label
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Convert frame to base64 for frontend
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            frame_results.append({
                'frame_number': frame_count,
                'image': frame_base64,
                'detections': len(results[0].boxes) if results[0].boxes is not None else 0
            })
        
        frame_count += 1
    
    cap.release()
    return {
        'frames': frame_results,
        'total_frames': frame_count,
        'processed_frames': len(frame_results)
    }

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