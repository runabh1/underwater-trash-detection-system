# 🌊 Underwater Trash Detection System

A beautiful web application for detecting underwater trash using your trained YOLO model. This system provides both video upload processing and live webcam detection capabilities with an immersive ocean-themed user interface.

## Features

- 🌊 **Ocean-themed UI**: Beautiful underwater design with animated bubbles and ocean gradients
- 📹 **Video Upload**: Upload and process underwater videos frame by frame
- 📷 **Live Webcam Detection**: Real-time trash detection using your device's camera
- 🔍 **Frame-by-Frame Analysis**: Detailed results showing each processed frame with detected trash
- 📊 **Grid Display**: Results displayed in an organized grid layout
- 🎯 **Bounding Box Visualization**: Green bounding boxes around detected trash items
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile devices

## Prerequisites

- Python 3.8 or higher
- Your trained YOLO model file (`best.pt`)
- Web browser with camera access (for live detection)

## Installation

1. **Clone or download this project** to your local machine

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your model file is in place**:
   - Make sure `best.pt` is in the root directory of the project

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Choose your detection method**:

   ### Video Upload
   - Click "Choose Video File" or drag and drop a video file
   - Supported formats: MP4, AVI, MOV
   - Click "Process Video" to analyze the video
   - View results in the grid below

   ### Live Webcam Detection
   - Click "Start Webcam" to begin live detection
   - Allow camera access when prompted
   - Real-time detection results will appear on the video feed
   - Click "Stop Webcam" to end the session

## How It Works

### Video Processing
1. Upload a video file through the web interface
2. The system processes every 5th frame to optimize performance
3. Your YOLO model detects trash in each frame
4. Bounding boxes are drawn around detected objects
5. Results are displayed in a grid format with frame numbers and detection counts

### Live Detection
1. Start the webcam through the browser
2. Frames are captured and sent to the server every second
3. Your model processes each frame in real-time
4. Detection results are overlaid on the video feed
5. Green bounding boxes show detected trash with confidence scores

## File Structure

```
under_trash/
├── app.py                 # Main Flask application
├── best.pt               # Your trained YOLO model
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── templates/
│   └── index.html       # Main web interface
└── uploads/             # Temporary upload directory (created automatically)
```

## Technical Details

- **Backend**: Flask web server with OpenCV and YOLO integration
- **Frontend**: HTML5, CSS3, JavaScript with modern web APIs
- **Model**: Uses Ultralytics YOLO for object detection
- **Video Processing**: OpenCV for frame extraction and processing
- **Real-time**: WebRTC for webcam access and real-time processing

## Customization

### Model Configuration
- The system automatically loads your `best.pt` model file
- Detection confidence thresholds can be adjusted in `app.py`
- Frame processing frequency can be modified (currently every 5th frame)

### UI Customization
- Ocean theme colors can be modified in the CSS section of `templates/index.html`
- Animation speeds and effects can be adjusted
- Grid layout and card styling can be customized

## Troubleshooting

### Common Issues

1. **Model not loading**:
   - Ensure `best.pt` is in the correct location
   - Check that the file is not corrupted

2. **Webcam not working**:
   - Ensure your browser supports WebRTC
   - Check camera permissions in your browser
   - Try using HTTPS (required for webcam in some browsers)

3. **Video upload fails**:
   - Check file format (MP4, AVI, MOV supported)
   - Ensure file size is reasonable
   - Check server logs for specific error messages

4. **Performance issues**:
   - Reduce video resolution for faster processing
   - Increase frame skip interval in `app.py`
   - Use a more powerful machine for large videos

### Error Messages

- **"No video file provided"**: Ensure you've selected a video file
- **"Model loading error"**: Check that `best.pt` exists and is valid
- **"Webcam access denied"**: Grant camera permissions in your browser

## Performance Tips

- For large videos, consider processing in smaller segments
- Adjust the frame processing interval based on your needs
- Use GPU acceleration if available (requires CUDA setup)
- Optimize video resolution before upload for faster processing

## Contributing

Feel free to enhance this system with additional features:
- Export detection results to CSV/JSON
- Video download with detection overlays
- Multiple model support
- Advanced filtering options
- Database integration for result storage

## License

This project is open source. Feel free to modify and distribute as needed.

---

**Happy detecting! 🌊♻️** 