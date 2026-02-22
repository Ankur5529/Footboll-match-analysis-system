# ⚽ Football Match Analysis System (AI Powered)

A complete, end-to-end artificial intelligence video tracking system and sleek Web Dashboard designed to analyze live football (soccer) match footage. It accurately tracks players, teams, the football, and referees, while calculating highly advanced tactical metrics and physical performance data frame-by-frame. 

All algorithms, models, and interfaces were built by **Ankur5529**.

---

## ⚡ Core Concepts & AI Tracking Pipeline

This application pipelines multiple powerful computer vision architectures into one seamless web application.

1. **High-Performance Object Detection (YOLO)**: Tracks individual players, referees, and the ball perfectly across complex sports footage.
2. **Auto-Clustering Team Assignments (KMeans)**: Automatically extracts bounding box pixel data from players, runs `KMeans Pixel Segmentation`, and groups players dynamically by their respective jersey colors, allowing the system to work on *any* two arbitrary teams.
3. **Dynamic Soccer Pitch View Transformation**: Maps 2D camera geometry to true 3D physical pitch dimensions in meters using mathematical perspective transformations.
4. **Camera Movement Estimation**: Calculates absolute object shifts using Optical Flow (`calcOpticalFlowPyrLK`), rendering actual geometry independent of camera panning, shaking, or zoom tracking.

---

## 📸 Application Flow & Dashboard Features

The application provides a fully reactive and asynchronous user-experience through a Flask-powered backend.

### 1. Main Page: Upload & Initialization
The primary interface provides a beautifully glass-morphed drag-and-drop zone. Users can easily upload `.mp4`, `.mov`, or `.avi` files natively through their browser. 

![Upload Page Interface](images/Screenshot%202026-02-23%20012701.png)

### 2. Live Processing Engine & Asynchronous Loading
Once a video is uploaded, the Flask server utilizes a background threading system to instantly begin processing. The UI prevents hanging by polling the server dynamically to track which stage of the Python pipeline is executing:
* **Reading** -> **Tracking Objects** -> **Camera Motion Array Calculation** -> **Tactics/Clustering** -> **OpenCV Rendering**.

![Processing & Loading Page](images/Screenshot%202026-02-23%20012826.png)

### 3. Analytics Dashboard & Physical Results
Upon completion, the system transitions into an interactive statistical engine! Using the geometrical transformations mapped earlier, the dashboard outputs three critical metrics calculated over the span of the video:
- ⏱️ **Total Ball Possession (%)**: The exact percentage of frames each team physically controlled the football.
- 📏 **Total Distance Covered (m)**: The cumulative distance tracked across the pitch for all detected field players per team.
- 🚀 **Max Sprint Speed (km/h)**: The absolute maximum velocity registered by any single player from that team.

Each statistic features beautifully responsive hover transitions and CSS graphs dynamically assigned to the exact Hex color code clustered from the AI team tracking.

![Analytics Graph Results](images/Screenshot%202026-02-23%20015802.png)

### 4. Video Object Overlays
Finally, the resulting encoded video generates directly above the statistics, allowing users to watch the entire process natively on the site, or download it! Players are tracked entirely with Team-Colored ellipses, and the ball is tracked with precise triangles.

![Live Pitch Overlays 1](images/Screenshot%202026-02-23%20020954.png)
![Live Pitch Overlays 2](images/Screenshot%202026-02-23%20021004.png)

---

## 🚀 How to Run the Project

### 1. Requirements & Setup
Ensure you are running **Python 3.x**. Inside your terminal run:
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Interface
Start the backend rendering server:
```bash
python app.py
```
Open a browser and navigate to `http://localhost:5000/`. Simply drag and drop any `.mp4` football clip into the uploader and wait for the dashboard to generate your possession and physics graphics!

### 3. Run Directly From Terminal (Headless Mode)
You can directly bypass the Web UI and analyze videos locally through the raw Python console utilizing your systems heavy cores. It saves out to a pre-defined path natively. 

```bash
python main.py -i "C:/path/to/my_match_clip.mp4" -o "C:/output/final_analytics_render.mp4"
```

---

*Repository 100% architected, maintained, and published by Ankur5529.*