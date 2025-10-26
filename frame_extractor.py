import cv2
import os
from pathlib import Path


class FrameExtractor:
    def __init__(self, video_path, output_dir="frames"):
        """
        Initialize frame extractor

        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted frames
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def extract_frames(self):
        """
        Extract all frames from video and save with index numbers

        Returns:
            tuple: (frame_count, fps, width, height)
        """
        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties:")
        print(f"  FPS: {fps}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Total frames: {frame_count}")
        print(f"\nExtracting frames...")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = self.output_dir / f"frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_idx += 1

            if frame_idx % 50 == 0:
                print(f"  Extracted {frame_idx} frames...")

        cap.release()
        print(f"✓ Extracted {frame_idx} frames to {self.output_dir}")

        return frame_idx, fps, width, height


if __name__ == "__main__":
    extractor = FrameExtractor("jumbled_video.mp4")
    count, fps, w, h = extractor.extract_frames()
    print(f"\nExtraction complete: {count} frames at {fps} FPS ({w}x{h})")
