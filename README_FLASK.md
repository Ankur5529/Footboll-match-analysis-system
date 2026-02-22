# Football Analysis Web App

## Overview
This is a Flask-based web interface for the Football Analysis project. It allows users to upload match footage, processes it using the AI tracking algorithms, and provides the analyzed output video.

## Features
- **Drag & Drop Upload**: Easy video upload interface.
- **Real-time Progress**: Visual feedback on each step of the analysis pipeline (Reading, Tracking, Camera Movement, Tactics, Rendering).
- **Football Themed UI**: Dark mode with neon green accents and glassmorphism.
- **Background Processing**: Handles long-running video processing tasks without freezing the UI.

## How to Run

1.  **Install Dependencies** (if not already installed):
    ```bash
    pip install flask
    ```
    (Ensure other project dependencies like `ultralytics`, `opencv-python`, etc., are also installed).

2.  **Start the Server**:
    Run the following command in your terminal:
    ```bash
    python app.py
    ```

3.  **Access the App**:
    Open your browser and navigate to:
    `http://localhost:5000`

## Structure
- `app.py`: The Flask backend server.
- `processor.py`: The video processing logic (refactored from main.py).
- `templates/index.html`: The frontend user interface.
- `uploads/`: Temporary storage for uploaded videos.
- `output_videos/`: Storage for processed videos.
