import cv2
from pathlib import Path
from tqdm import tqdm


class VideoWriter:
    def __init__(self, output_path="output/reconstructed.mp4", fps=30):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(exist_ok=True)
        self.fps = fps

    def write_video(self, frames_data, sequence):
        print(f"\nWriting reconstructed video...")
        print(f"  Output: {self.output_path}")
        print(f"  FPS: {self.fps}")
        print(f"  Frames: {len(sequence)}")

        first_frame = frames_data[sequence[0]]["image"]
        height, width = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (width, height))

        if not out.isOpened():
            raise ValueError("Could not open video writer")

        for frame_idx in tqdm(sequence, desc="Writing frames"):
            frame = frames_data[frame_idx]["image"]
            out.write(frame)

        out.release()
        print(f"✓ Video written successfully")

    def create_side_by_side_comparison(
        self,
        frames_data,
        original_order,
        reconstructed_order,
        output_path="output/comparison.mp4",
    ):
        print(f"\nCreating comparison video...")

        first_frame = frames_data[original_order[0]]["image"]
        height, width = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width * 2, height))

        for orig_idx, recon_idx in tqdm(
            zip(original_order, reconstructed_order),
            total=len(original_order),
            desc="Creating comparison",
        ):
            orig_frame = frames_data[orig_idx]["image"]
            recon_frame = frames_data[recon_idx]["image"]

            combined = cv2.hconcat([orig_frame, recon_frame])

            cv2.putText(
                combined,
                "Original Order",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                combined,
                "Reconstructed",
                (width + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            out.write(combined)

        out.release()
        print(f"✓ Comparison video created: {output_path}")


if __name__ == "__main__":
    print("Video writer module - use through main pipeline")
