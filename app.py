# app.py
"""
Flask web application for Classroom Safety Narrator
Provides web interface for real-time classroom monitoring
"""

import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from PIL import Image
import io
import threading
import time
from collections import deque

from narrator_main import Narrator

# Configuration constants
STATE_COLORS = {
    'normal class activity': '#4CAF50',
    'lone child may need attention': '#FF9800',
    'dangerous object near child': '#F44336',
    'empty classroom': '#9E9E9E'
}
DEFAULT_STATE_COLOR = '#2196F3'
MAX_HISTORY = 50
FRAME_SKIP = 2
VIDEO_FPS = 30
JPEG_QUALITY = 85

app = Flask(__name__)
CORS(app)

# Global variables for video capture
narrator = None
camera = None
camera_lock = threading.Lock()
is_streaming = False
frame_skip_counter = 0

# Global state storage for frontend
current_state = {
    'state_text': 'Waiting...',
    'state_color': '#9E9E9E',
    'narration': 'Waiting for analysis...',
    'detections': [],
    'last_update': 0
}
state_lock = threading.Lock()

# History log storage (uses deque for efficient append/pop)
history_log = deque(maxlen=MAX_HISTORY)
history_lock = threading.Lock()

# Initialize narrator (loads YOLO model)
def init_narrator():
    global narrator
    if narrator is None:
        print("[INFO] Initializing Narrator...")
        narrator = Narrator()
        print("[INFO] Narrator ready!")
    return narrator

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    """Start camera capture"""
    global camera, is_streaming
    
    data = request.get_json() or {}
    camera_index = data.get('camera_index', 1)
    
    # Validate camera index
    try:
        camera_index = int(camera_index)
        if camera_index < 0 or camera_index > 10:  # Reasonable range
            return jsonify({'success': False, 'error': f'Invalid camera index: {camera_index}. Must be 0-10.'}), 400
    except (ValueError, TypeError):
        return jsonify({'success': False, 'error': 'Camera index must be a number'}), 400
    
    with camera_lock:
        if camera is not None:
            camera.release()
        
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            return jsonify({'success': False, 'error': f'Cannot open camera {camera_index}'}), 400
        
        is_streaming = True
        return jsonify({'success': True, 'message': f'Camera {camera_index} started'})

@app.route('/api/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera capture"""
    global camera, is_streaming
    
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
        is_streaming = False
    
    # Reset state
    with state_lock:
        current_state['state_text'] = 'Waiting...'
        current_state['state_color'] = '#9E9E9E'
        current_state['narration'] = 'Waiting for analysis...'
        current_state['detections'] = []
        current_state['last_update'] = 0
    
    # Clear history when stopping
    with history_lock:
        history_log.clear()
    
    return jsonify({'success': True, 'message': 'Camera stopped'})

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """Process a single frame from the frontend"""
    global narrator, frame_skip_counter
    
    if narrator is None:
        init_narrator()
    
    try:
        # Get image data from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'error': 'Failed to decode image'}), 400
        
        # Process frame (with frame skipping)
        frame_skip_counter += 1
        if frame_skip_counter % FRAME_SKIP != 0:
            return jsonify({'success': True, 'skip': True})
        
        # Run detection and narration
        dets, narration, state_text = narrator.narrate(frame)
        
        # Draw bounding boxes on frame
        annotated_frame = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            label = f"{d['class']} {d['conf']:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Encode annotated frame to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        annotated_image = base64.b64encode(buffer).decode('utf-8')
        
        # Determine state color
        state_color = STATE_COLORS.get(state_text, DEFAULT_STATE_COLOR)
        
        # Use actual AI narration, only fallback if truly empty (shouldn't happen with bypass_suppression)
        narration_display = narration if narration and narration.strip() else f'Scene analysis: {state_text}.'
        
        # Add to history log (deque automatically maintains maxlen)
        with history_lock:
            history_log.append({
                'timestamp': time.time(),
                'state': state_text,
                'state_color': state_color,
                'narration': narration_display,
                'detections_count': len(dets),
                'detections': [
                    {
                        'class': d['class'],
                        'confidence': round(d['conf'], 2)
                    }
                    for d in dets[:5]  # Store top 5 detections
                ]
            })
        
        return jsonify({
            'success': True,
            'annotated_image': f'data:image/jpeg;base64,{annotated_image}',
            'detections': [
                {
                    'class': d['class'],
                    'confidence': round(d['conf'], 2),
                    'box': d['box']
                }
                for d in dets
            ],
            'narration': narration_display,
            'state': state_text,
            'state_color': state_color
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/video_feed')
def video_feed():
    """Video streaming route (MJPEG stream)"""
    global camera, narrator, is_streaming, frame_skip_counter
    
    if narrator is None:
        init_narrator()
    
    def generate():
        global frame_skip_counter
        frame_skip_counter = 0
        
        while is_streaming:
            with camera_lock:
                if camera is None or not camera.isOpened():
                    break
                
                ret, frame = camera.read()
                if not ret:
                    break
                
                frame_skip_counter += 1
                if frame_skip_counter % FRAME_SKIP != 0:
                    continue
                
                # Process frame
                try:
                    dets, narration, state_text = narrator.narrate(frame)
                    
                    # Update global state for frontend
                    state_color = STATE_COLORS.get(state_text, DEFAULT_STATE_COLOR)
                    
                    with state_lock:
                        current_state['state_text'] = state_text
                        current_state['state_color'] = state_color
                        # Use actual AI narration, only fallback if truly empty (shouldn't happen with bypass_suppression)
                        narration_display = narration if narration and narration.strip() else f'Scene analysis: {state_text}.'
                        current_state['narration'] = narration_display
                        current_state['detections'] = [
                            {
                                'class': d['class'],
                                'confidence': round(d['conf'], 2),
                                'box': d['box']
                            }
                            for d in dets
                        ]
                        current_state['last_update'] = time.time()
                    
                    # Add to history log (deque automatically maintains maxlen)
                    with history_lock:
                        history_log.append({
                            'timestamp': time.time(),
                            'state': state_text,
                            'state_color': state_color,
                            'narration': narration_display,
                            'detections_count': len(dets),
                            'detections': [
                                {
                                    'class': d['class'],
                                    'confidence': round(d['conf'], 2)
                                }
                                for d in dets[:5]  # Store top 5 detections
                            ]
                        })
                    
                    # Draw bounding boxes
                    for d in dets:
                        x1, y1, x2, y2 = d["box"]
                        label = f"{d['class']} {d['conf']:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 4),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    
                    # Add state overlay
                    header = f"[{state_text.upper()}] {narration[:100] if narration else 'Processing...'}"
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
                    cv2.putText(frame, header, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                except Exception as e:
                    print(f"[ERROR] Processing frame: {e}")
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(1.0 / VIDEO_FPS)  # Maintain target FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/current_state', methods=['GET'])
def get_current_state():
    """Get the current safety state from video feed processing"""
    with state_lock:
        return jsonify({
            'success': True,
            'state': current_state['state_text'],
            'state_color': current_state['state_color'],
            'narration': current_state['narration'],
            'detections': current_state['detections'],
            'last_update': current_state['last_update']
        })

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get the history log of narrations and states"""
    with history_lock:
        # Return history in reverse chronological order (newest first)
        # Convert deque to list and reverse
        history_list = list(history_log)
        history_list.reverse()
        return jsonify({
            'success': True,
            'history': history_list,
            'count': len(history_list)
        })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'narrator_loaded': narrator is not None,
        'camera_active': camera is not None and camera.isOpened() if camera else False
    })

if __name__ == '__main__':
    # Initialize narrator on startup
    init_narrator()
    
    print("\n" + "="*60)
    print("Classroom Safety Narrator - Web Interface")
    print("="*60)
    print("\nStarting Flask server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Use environment variable for debug mode (default: False for production)
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.run(host='0.0.0.0', port=5000, debug=debug_mode, threaded=True)

