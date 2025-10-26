import cv2
import os
from pathlib import Path


def validate_video_specs(video_path, expected_duration=10, expected_fps=30):
    """
    Validate video specifications match expected parameters

    Args:
        video_path: Path to video file
        expected_duration: Expected duration in seconds
        expected_fps: Expected frames per second

    Returns:
        dict: Video properties
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    props = {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }

    print("\nVideo Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Frame Count: {frame_count}")
    print(f"  Duration: {duration:.2f} seconds")

    expected_frames = expected_duration * expected_fps
    if abs(frame_count - expected_frames) > 5:
        print(f"\n⚠ WARNING: Expected ~{expected_frames} frames, got {frame_count}")
        print(f"  This may indicate incorrect video specifications")

    return props


def check_disk_space(required_gb=5):
    """
    Check available disk space

    Args:
        required_gb: Required space in GB

    Returns:
        bool: True if sufficient space
    """
    import shutil

    stat = shutil.disk_usage(".")
    available_gb = stat.free / (1024**3)

    print(f"\nDisk Space:")
    print(f"  Available: {available_gb:.2f} GB")
    print(f"  Required: {required_gb:.2f} GB")

    if available_gb < required_gb:
        print(f"  ⚠ WARNING: Low disk space!")
        return False

    return True


def estimate_memory_usage(frame_count, width, height):
    frame_size_mb = (width * height * 3) / (1024**2)

    frames_memory = (frame_size_mb * frame_count) / 1024

    matrix_memory = (frame_count**2 * 2) / (1024**3)

    total_memory = (frames_memory + matrix_memory) * 1.2

    print(f"\nEstimated Memory Usage:")
    print(f"  Frames: {frames_memory:.2f} GB")
    print(f"  Similarity Matrix: {matrix_memory:.3f} GB")
    print(f"  Total (with overhead): {total_memory:.2f} GB")

    return total_memory


def create_sample_visualization(
    frames_data, sequence, output_path="output/sample_frames.jpg"
):
    """
    Create visualization showing first 10 reconstructed frames

    Args:
        frames_data: Dictionary of frame data
        sequence: Reconstructed sequence
        output_path: Output image path
    """
    import numpy as np

    print(f"\nCreating sample visualization...")

    sample_size = min(10, len(sequence))
    samples = []

    for i in range(sample_size):
        frame = frames_data[sequence[i]]["image"]
        thumbnail = cv2.resize(frame, (192, 108))

        cv2.putText(
            thumbnail,
            f"#{i + 1}",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        samples.append(thumbnail)

    row1 = cv2.hconcat(samples[:5])
    row2 = cv2.hconcat(samples[5:10] if sample_size > 5 else samples[5:])

    if sample_size < 10:
        padding = np.zeros((108, 192 * (10 - sample_size), 3), dtype=np.uint8)
        row2 = cv2.hconcat([row2, padding]) if row2.size > 0 else padding

    visualization = cv2.vconcat([row1, row2])

    cv2.imwrite(output_path, visualization)
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    print("Utility functions module")
