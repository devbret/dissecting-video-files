import os
import cv2
import numpy as np
import json
import math

input_dir = 'videos'
os.makedirs(input_dir, exist_ok=True)

frame_skip = 1
frame_resize = (320, 240)

video_metrics = {}

def convert_numpy(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def calc_entropy(gray_frame):
    hist, _ = np.histogram(gray_frame, bins=256, range=(0, 256))
    total_pixels = np.sum(hist)
    if total_pixels == 0:
        return 0.0
    p = hist / total_pixels
    entropy = 0.0
    for val in p:
        if val > 0:
            entropy -= val * math.log2(val)
    return entropy

for video_file in os.listdir(input_dir):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(input_dir, video_file)

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read '{video_file}'. Skipping.")
            cap.release()
            continue

        prev_frame_resized = cv2.resize(frame, frame_resize)
        prev_gray = cv2.cvtColor(prev_frame_resized, cv2.COLOR_BGR2GRAY)

        fgbg = cv2.createBackgroundSubtractorMOG2()

        frames_data = []
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_index = 0

        while ret:
            for _ in range(frame_skip - 1):
                ret, frame = cap.read()
                frame_index += 1
                if not ret:
                    break

            if not ret:
                break

            curr_frame_resized = cv2.resize(frame, frame_resize)
            curr_gray = cv2.cvtColor(curr_frame_resized, cv2.COLOR_BGR2GRAY)

            current_time = frame_index / frame_rate

            fgmask = fgbg.apply(curr_gray)
            motion_score = np.sum(fgmask) / 255.0

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_magnitude = np.linalg.norm(flow, axis=2)
            flow_score = np.mean(flow_magnitude)

            avg_luminance = np.mean(curr_gray)

            laplacian_var = cv2.Laplacian(curr_gray, cv2.CV_64F).var()

            edges = cv2.Canny(curr_gray, 100, 200)
            edge_count = np.count_nonzero(edges)

            b, g, r = cv2.split(curr_frame_resized)

            avg_red = np.mean(r)

            avg_green = np.mean(g)

            avg_blue = np.mean(b)

            hsv = cv2.cvtColor(curr_frame_resized, cv2.COLOR_BGR2HSV)
            h_channel, s_channel, v_channel = cv2.split(hsv)

            avg_hue = np.mean(h_channel)

            avg_saturation = np.mean(s_channel)

            grayscale_contrast = float(curr_gray.max() - curr_gray.min())

            grayscale_stddev = float(np.std(curr_gray))

            grayscale_entropy = calc_entropy(curr_gray)

            saturation_stddev = float(np.std(s_channel))

            hue_stddev = float(np.std(h_channel))

            red_stddev = float(np.std(r))

            green_stddev = float(np.std(g))

            blue_stddev = float(np.std(b))

            sobelx = cv2.Sobel(curr_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(curr_gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)

            sobel_magnitude_mean = float(np.mean(sobel_magnitude))

            sobel_magnitude_std = float(np.std(sobel_magnitude))

            frame_data = {
                "time": current_time,
                "motion_score": float(motion_score),
                "flow_score": float(flow_score),
                "avg_luminance": float(avg_luminance),
                "laplacian_var": float(laplacian_var),
                "edge_count": int(edge_count),
                "avg_red": float(avg_red),
                "avg_green": float(avg_green),
                "avg_blue": float(avg_blue),
                "avg_hue": float(avg_hue),
                "avg_saturation": float(avg_saturation),
                "grayscale_contrast": grayscale_contrast,
                "grayscale_stddev": grayscale_stddev,
                "grayscale_entropy": grayscale_entropy,
                "saturation_stddev": saturation_stddev,
                "hue_stddev": hue_stddev,
                "red_stddev": red_stddev,
                "green_stddev": green_stddev,
                "blue_stddev": blue_stddev,
                "sobel_magnitude_mean": sobel_magnitude_mean,
                "sobel_magnitude_std": sobel_magnitude_std
            }

            frames_data.append(frame_data)

            prev_gray = curr_gray
            ret, frame = cap.read()
            frame_index += 1

        cap.release()
        video_metrics[video_file] = frames_data

output_json = 'video_metrics.json'
with open(output_json, 'w') as f:
    json.dump(video_metrics, f, indent=4, default=convert_numpy)

print(f"Frame-by-frame metrics have been saved to '{output_json}'.")

