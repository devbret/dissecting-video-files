import os
import cv2
import numpy as np
import ffmpeg
import json

input_dir = 'videos'
output_dir = 'scenes'
os.makedirs(output_dir, exist_ok=True)

motion_threshold = 10000
min_scene_duration = 2 
frame_skip = 3
frame_resize = (320, 240)

metrics = {}

def convert_numpy(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

for video_file in os.listdir(input_dir):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(input_dir, video_file)

        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()
        if not ret:
            print(f"Failed to read {video_file}")
            continue

        prev_frame = cv2.resize(prev_frame, frame_resize)
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        scene_start = 0
        scene_counter = 1

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        last_scene_time = 0

        video_metrics = []

        while ret:
            for _ in range(frame_skip):
                ret, curr_frame = cap.read()
                if not ret:
                    break
            if not ret:
                break

            curr_frame_resized = cv2.resize(curr_frame, frame_resize)
            curr_frame_gray = cv2.cvtColor(curr_frame_resized, cv2.COLOR_BGR2GRAY)

            frame_diff = cv2.absdiff(prev_frame, curr_frame_gray)
            motion_score = np.sum(frame_diff)

            current_time = frame_count / frame_rate

            video_metrics.append({
                'time': current_time,
                'motion_score': float(motion_score)
            })

            if motion_score > motion_threshold and (current_time - last_scene_time) > min_scene_duration:
                scene_end = current_time

                output_file = os.path.join(output_dir, f'{os.path.splitext(video_file)[0]}_scene_{scene_counter}.mp4')
                try:
                    (
                        ffmpeg.input(video_path, ss=scene_start, to=scene_end)
                        .output(output_file, vcodec='copy', acodec='copy')
                        .run(overwrite_output=True)
                    )
                    print(f"Scene {scene_counter} saved: {output_file}")
                except Exception as e:
                    print(f"Error processing scene {scene_counter}: {e}")

                scene_start = scene_end
                last_scene_time = current_time
                scene_counter += 1

            prev_frame = curr_frame_gray
            frame_count += frame_skip

        scene_end = cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_rate
        if scene_end - scene_start > min_scene_duration:
            output_file = os.path.join(output_dir, f'{os.path.splitext(video_file)[0]}_scene_{scene_counter}.mp4')
            try:
                (
                    ffmpeg.input(video_path, ss=scene_start, to=scene_end)
                    .output(output_file, vcodec='copy', acodec='copy')
                    .run(overwrite_output=True)
                )
                print(f"Scene {scene_counter} saved: {output_file}")
            except Exception as e:
                print(f"Error processing scene {scene_counter}: {e}")

        metrics[video_file] = video_metrics
        cap.release()

with open('video_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4, default=convert_numpy)

print("Motion-based scene splitting complete! Metrics saved to video_metrics.json")
