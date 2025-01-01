import cv2
import os

VIDEO_PATH = "path/to/your/video.mp4"
OUTPUT_DIR = "frames"
FRAME_PREFIX = "frame"
IMG_FORMAT = "jpg"

def extract_frames(video_path, output_dir, prefix="frame", img_format="jpg"):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_paths = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_filename = f"{prefix}_{str(frame_index).zfill(6)}.{img_format}"
        output_filepath = os.path.join(output_dir, output_filename)

        cv2.imwrite(output_filepath, frame)

        frame_paths.append(output_filepath)

        frame_index += 1

    cap.release()
    return frame_paths


def main():
    print(f"Extracting frames from: {VIDEO_PATH}")
    print(f"Saving frames to: {OUTPUT_DIR}")

    frame_paths = extract_frames(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        prefix=FRAME_PREFIX,
        img_format=IMG_FORMAT
    )

    print(f"Extraction complete: {len(frame_paths)} frames saved.")
    for path in frame_paths[:5]:
        print(f"  Sample frame path: {path}")


if __name__ == "__main__":
    main()
