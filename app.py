import os
import uuid
import threading
import time
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from processor import process_video

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output_videos')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Global dictionary to store task status
# Structure: task_id -> {'status': str, 'progress': int, 'message': str, 'output_file': str}
tasks = {}

def update_task_progress(task_id, message, progress):
    tasks[task_id]['message'] = message
    tasks[task_id]['progress'] = progress

def run_processing(task_id, input_path, output_filename):
    try:
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Define a callback to update progress
        def progress_callback(step_name, percent):
            update_task_progress(task_id, step_name, percent)
            
        # Run the processing
        output_path, stats = process_video(input_path, output_path, MODEL_PATH, progress_callback)
        
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['output_file'] = output_filename
        tasks[task_id]['stats'] = stats
        tasks[task_id]['progress'] = 100
        tasks[task_id]['message'] = "Processing complete!"
        
    except Exception as e:
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['message'] = str(e)
        print(f"Error processing video: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = file.filename
        unique_filename = f"{uuid.uuid4()}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(input_path)
        
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting...',
            'output_file': None
        }
        
        # Start processing in a background thread
        # Force .mp4 extension for output
        output_filename = f"processed_{uuid.uuid4()}_{os.path.splitext(filename)[0]}.mp4"
        thread = threading.Thread(target=run_processing, args=(task_id, input_path, output_filename))
        thread.start()
        
        return jsonify({'task_id': task_id})

@app.route('/status/<task_id>')
def task_status(task_id):
    task = tasks.get(task_id)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Task not found'}), 404

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
