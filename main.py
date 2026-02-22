import os
import argparse
from processor import process_video

def main():
    parser = argparse.ArgumentParser(description="Process football videos for analysis.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to input video file")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to save output video file")
    
    args = parser.parse_args()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')
    
    if not os.path.exists(args.input):
        print(f"Error: Input video not found at: {args.input}")
        return
        
    print(f"Input Video: {args.input}")
    print(f"Output Video: {args.output}")
    print(f"Model Path: {MODEL_PATH}")
    print("-" * 40)
    
    def progress_callback(step_name, percent):
        print(f"[{percent}%] {step_name}")
        
    try:
        output_path, stats = process_video(args.input, args.output, MODEL_PATH, progress_callback)
        print("\n" + "="*40)
        print("FINAL ANALYSIS STATS")
        print("="*40)
        print(f"Team 1 Possession: {stats['team_1']['possession']}% | Team 1 Color: {stats['team_1']['color']}")
        print(f"   Max Speed: {stats['team_1']['max_speed']} km/h | Distance: {stats['team_1']['total_distance']} m")
        print("-" * 40)
        print(f"Team 2 Possession: {stats['team_2']['possession']}% | Team 2 Color: {stats['team_2']['color']}")
        print(f"   Max Speed: {stats['team_2']['max_speed']} km/h | Distance: {stats['team_2']['total_distance']} m")
        print("="*40)
        print(f"\nAnalysis complete! Video saved successfully at: {output_path}")
        
    except Exception as e:
        print(f"Processing failed: {e}")

if __name__ == '__main__':
    main()