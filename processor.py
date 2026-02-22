import os
import cv2
import numpy as np
import sys
import subprocess
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def process_video(input_path, output_path, model_path, progress_callback=None):
    """
    Process a football video and save the result.
    progress_callback: A function that accepts a string (step name) and an integer (0-100).
    """

    def update_progress(step_name, percent):
        if progress_callback:
            progress_callback(step_name, percent)

    update_progress("Initializing...", 0)

    # Validate paths
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Read Video
    update_progress("Reading video...", 5)
    video_frames, fps = read_video(input_path)
    
    # Initialize Tracker
    update_progress("Initializing tracker...", 10)
    tracker = Tracker(model_path)

    # Get object tracks
    # Note: We are NOT using stubs here to ensure fresh processing for web uploads
    update_progress("Tracking objects...", 15)
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False)
    
    update_progress("Adding positions to tracks...", 40)
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    update_progress("Estimating camera movement...", 45)
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=False)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    update_progress("Transforming view...", 60)
    frame_h, frame_w = video_frames[0].shape[:2]
    view_transformer = ViewTransformer(frame_width=frame_w, frame_height=frame_h)
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    update_progress("Interpolating ball positions...", 65)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    update_progress("Calculating speed and distance...", 70)
    speed_and_distance_estimator = SpeedAndDistance_Estimator(fps=fps)
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    update_progress("Assigning player teams...", 75)
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors.get(team, (0, 0, 255))

    # Assign Ball Acquisition
    update_progress("Assigning ball acquisition...", 85)
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        if 1 not in tracks['ball'][frame_num] or 'bbox' not in tracks['ball'][frame_num][1]:
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(0)
            continue
            
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(0)
    team_ball_control = np.array(team_ball_control)

    # Draw output 
    update_progress("Drawing annotations...", 90)
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save video with FFmpeg Conversion for Browser Compatibility
    update_progress("Saving and Converting Video...", 95)
    
    # Save to a temporary AVI file first (since save_video uses XVID)
    temp_output_path = output_path.replace('.mp4', '_temp.avi')
    if temp_output_path == output_path:
        temp_output_path += "_temp.avi"
        
    save_video(output_video_frames, temp_output_path, fps=fps)
    
    # Convert to browser-compatible MP4 (H.264) using FFmpeg
    try:
        command = [
            'ffmpeg', '-y', 
            '-i', temp_output_path, 
            '-vcodec', 'libx264', 
            '-preset', 'fast',    # Faster encoding
            '-crf', '23',         # Standard quality
            '-acodec', 'aac', 
            '-movflags', '+faststart', # Enable web streaming
            output_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Cleanup temp file
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}")
        # Fallback: Just rename the temp file if conversion fails (better than nothing)
        if os.path.exists(temp_output_path):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_output_path, output_path)
    
    update_progress("Computing Stats...", 98)
    team_1_frames = float(np.sum(team_ball_control == 1))
    team_2_frames = float(np.sum(team_ball_control == 2))
    total_frames_with_ball = team_1_frames + team_2_frames

    possession_team_1 = team_1_frames / total_frames_with_ball if total_frames_with_ball > 0 else 0
    possession_team_2 = team_2_frames / total_frames_with_ball if total_frames_with_ball > 0 else 0

    def bgr_to_hex(bgr):
        return f"#{int(bgr[2]):02x}{int(bgr[1]):02x}{int(bgr[0]):02x}"
        
    team_1_max_speed = 0.0
    team_2_max_speed = 0.0
    player_distances = {1: {}, 2: {}} 

    for frame_num in range(len(tracks['players'])):
        for track_id, track_info in tracks['players'][frame_num].items():
            team = track_info.get('team')
            if team not in (1, 2):
                continue
            
            speed = track_info.get('speed', 0.0)
            distance = track_info.get('distance', 0.0)
            
            if team == 1:
                if speed > team_1_max_speed:
                    team_1_max_speed = speed
                if track_id not in player_distances[1] or distance > player_distances[1][track_id]:
                    player_distances[1][track_id] = distance
            elif team == 2:
                if speed > team_2_max_speed:
                    team_2_max_speed = speed
                if track_id not in player_distances[2] or distance > player_distances[2][track_id]:
                    player_distances[2][track_id] = distance

    team_1_total_distance = sum(player_distances[1].values())
    team_2_total_distance = sum(player_distances[2].values())

    stats = {
        'team_1': {
            'possession': round(possession_team_1 * 100, 2),
            'color': bgr_to_hex(team_assigner.team_colors.get(1, [255, 0, 0])),
            'max_speed': round(team_1_max_speed, 2),
            'total_distance': round(team_1_total_distance, 2)
        },
        'team_2': {
            'possession': round(possession_team_2 * 100, 2),
            'color': bgr_to_hex(team_assigner.team_colors.get(2, [0, 0, 255])),
            'max_speed': round(team_2_max_speed, 2),
            'total_distance': round(team_2_total_distance, 2)
        }
    }

    update_progress("Done!", 100)
    return output_path, stats
