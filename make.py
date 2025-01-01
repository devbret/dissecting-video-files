import os
import json
import shutil
import subprocess
import random

AUDIO_JSON_PATH   = "frame_aligned_audio.json"
VIDEO_JSON_PATH   = "video_metrics.json"
FRAMES_DIR        = "frames"
AUDIO_FILE        = "path/to/your/audio.mp3"
OUTPUT_MP4        = "remixed_output.mp4"
TEMP_FRAMES_DIR   = "temp_frames"

FPS = 60


TOP_K = 200

AVOID_REPEAT_FRAMES = True

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def pick_diverse_frame_for_moment(audio_metrics, video_frames, used_frames, top_k=20):
    audio_loudness = audio_metrics.get("loudness", 0.0)

    scored = []
    for i, vf in enumerate(video_frames):
        motion = vf.get("motion_score", 0.0)
        score = audio_loudness * motion
        scored.append((i, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    top_k_candidates = scored[:top_k]

    if AVOID_REPEAT_FRAMES:
        filtered = [(idx, sc) for (idx, sc) in top_k_candidates if idx not in used_frames]
        if len(filtered) > 0:
            top_k_candidates = filtered

    if len(top_k_candidates) == 0:
        chosen_index = scored[0][0]
    else:
        chosen_index = random.choice(top_k_candidates)[0]

    return chosen_index


def build_frame_sequence(audio_data, video_data):
    first_vid_key = next(iter(video_data.keys()))
    video_frames = video_data[first_vid_key]
    video_frames.sort(key=lambda x: x["time"])

    aligned_audio = audio_data["aligned_audio_data"]
    aligned_audio.sort(key=lambda x: x["time"])

    final_sequence = []

    used_frames = set()

    for new_seq_index, audio_metrics in enumerate(aligned_audio):
        chosen_video_index = pick_diverse_frame_for_moment(audio_metrics, video_frames, used_frames, TOP_K)

        final_sequence.append((new_seq_index, chosen_video_index))

        if AVOID_REPEAT_FRAMES:
            used_frames.add(chosen_video_index)

    return final_sequence


def prepare_frames_in_order(frame_sequence, frames_dir, temp_dir):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    for (new_seq_idx, old_idx) in frame_sequence:
        old_filename = f"frame_{old_idx:06d}.jpg"
        old_path = os.path.join(frames_dir, old_filename)

        new_filename = f"frame_{new_seq_idx+1:06d}.jpg"
        new_path = os.path.join(temp_dir, new_filename)

        if not os.path.exists(old_path):
            print(f"Warning: missing {old_path}")
            continue

        try:
            os.link(old_path, new_path)
        except OSError:
            shutil.copy2(old_path, new_path)


def assemble_video_with_audio(frames_dir, audio_file, fps, output_file):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%06d.jpg"),
        "-i", audio_file,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        output_file
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    audio_data = load_json(AUDIO_JSON_PATH)
    video_data = load_json(VIDEO_JSON_PATH)

    frame_sequence = build_frame_sequence(audio_data, video_data)

    prepare_frames_in_order(frame_sequence, FRAMES_DIR, TEMP_FRAMES_DIR)

    assemble_video_with_audio(TEMP_FRAMES_DIR, AUDIO_FILE, FPS, OUTPUT_MP4)

    print(f"Done! Created '{OUTPUT_MP4}' with an audio track and diverse frames.")


if __name__ == "__main__":
    random.seed()
    main()
