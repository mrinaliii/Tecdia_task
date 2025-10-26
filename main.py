import time
import argparse
from pathlib import Path
import json

from frame_extractor import FrameExtractor
from frame_loader import FrameLoader
from frame_reconstructor import FrameReconstructor
from video_writer import VideoWriter
from similarity_metrics import SimilarityCalculator
from utils import (
    validate_video_specs,
    check_disk_space,
    estimate_memory_usage,
    create_sample_visualization,
)


class ExecutionLogger:
    def __init__(self, log_file="execution_log.txt"):
        self.log_file = log_file
        self.timings = {}
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def log_step(self, step_name, duration):
        self.timings[step_name] = duration
        print(f"  [{step_name}] completed in {duration:.2f} seconds")

    def finish_and_save(self):
        total_time = time.time() - self.start_time
        self.timings["total"] = total_time

        report = []
        report.append("=" * 60)
        report.append("JUMBLED FRAMES RECONSTRUCTION - EXECUTION LOG")
        report.append("=" * 60)
        report.append(f"\nTotal Execution Time: {total_time:.2f} seconds")
        report.append(f"                      ({total_time / 60:.2f} minutes)")
        report.append("\nStep-by-Step Breakdown:")
        report.append("-" * 60)

        for step, duration in self.timings.items():
            if step != "total":
                percentage = (duration / total_time) * 100
                report.append(f"  {step:.<45} {duration:>6.2f}s ({percentage:>5.1f}%)")

        report.append("=" * 60)

        with open(self.log_file, "w") as f:
            f.write("\n".join(report))

        print("\n" + "\n".join(report))

        json_file = self.log_file.replace(".txt", ".json")
        with open(json_file, "w") as f:
            json.dump(self.timings, f, indent=2)

        return total_time


def preflight_checks(video_path):
    print("\n" + "=" * 60)
    print("PRE-FLIGHT CHECKS")
    print("=" * 60)

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    props = validate_video_specs(video_path)

    check_disk_space(required_gb=5)

    memory_needed = estimate_memory_usage(
        props["frame_count"], props["width"], props["height"]
    )

    if memory_needed > 14:
        print("\n⚠ WARNING: Reconstruction may use significant memory")
        print("  Consider closing other applications")

    print("\n✓ Pre-flight checks complete\n")

    return props


def main(video_path, output_path="output/reconstructed.mp4", fps=30):
    video_props = preflight_checks(video_path)

    print("\n" + "=" * 60)
    print("JUMBLED FRAMES RECONSTRUCTION PIPELINE")
    print("=" * 60 + "\n")

    logger = ExecutionLogger()
    logger.start()
    print("\n" + "=" * 60)
    print("JUMBLED FRAMES RECONSTRUCTION PIPELINE")
    print("=" * 60 + "\n")

    logger = ExecutionLogger()
    logger.start()

    print("STEP 1: Frame Extraction")
    print("-" * 60)
    step_start = time.time()
    extractor = FrameExtractor(video_path)
    frame_count, video_fps, width, height = extractor.extract_frames()
    logger.log_step("Frame Extraction", time.time() - step_start)

    expected_frames = 10 * 30
    if frame_count != expected_frames:
        print(f"\n⚠ WARNING: Expected {expected_frames} frames, got {frame_count}")

    print("\nSTEP 2: Frame Loading & Hash Computation")
    print("-" * 60)
    step_start = time.time()
    loader = FrameLoader()
    frames_data = loader.load_all_frames_parallel(max_workers=10)
    logger.log_step("Frame Loading", time.time() - step_start)

    print("\nSTEP 3: Similarity Matrix Computation")
    print("-" * 60)
    step_start = time.time()
    similarity_matrix, frame_indices = loader.compute_hash_similarity_matrix(
        frames_data
    )
    logger.log_step("Similarity Matrix", time.time() - step_start)

    print("\nSTEP 4: Frame Sequence Reconstruction")
    print("-" * 60)
    step_start = time.time()
    reconstructor = FrameReconstructor(frames_data, similarity_matrix, frame_indices)
    sequence = reconstructor.greedy_nearest_neighbor()
    logger.log_step("Initial Reconstruction", time.time() - step_start)

    step_start = time.time()
    sequence = reconstructor.try_reverse_sequence(sequence)
    logger.log_step("Direction Testing", time.time() - step_start)

    print("\nSTEP 5: SSIM Refinement")
    print("-" * 60)
    step_start = time.time()
    sequence = reconstructor.refine_with_ssim(sequence, window_size=4)
    logger.log_step("SSIM Refinement", time.time() - step_start)

    print("\nSaving sequence...")
    save_sequence(sequence)

    print("\nSTEP 6: Video Writing")
    print("-" * 60)
    step_start = time.time()
    writer = VideoWriter(output_path, fps=fps)
    writer.write_video(frames_data, sequence)
    logger.log_step("Video Writing", time.time() - step_start)

    print("\nSTEP 7: Creating Sample Visualization")
    print("-" * 60)
    step_start = time.time()
    create_sample_visualization(frames_data, sequence)
    logger.log_step("Visualization", time.time() - step_start)

    print("\nSTEP 8: Quality Assessment")
    print("-" * 60)
    step_start = time.time()
    avg_similarity = calculate_reconstruction_quality(frames_data, sequence)
    logger.log_step("Quality Assessment", time.time() - step_start)

    total_time = logger.finish_and_save()

    print(f"\n{'=' * 60}")
    print("RECONSTRUCTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Output: {output_path}")
    print(f"  Frames: {len(sequence)}")
    print(f"  Average Frame Similarity: {avg_similarity:.2%}")
    print(f"  Total Time: {total_time:.2f} seconds")
    print(f"  Log saved to: execution_log.txt")
    print(f"{'=' * 60}\n")


def calculate_reconstruction_quality(frames_data, sequence):
    calc = SimilarityCalculator()

    print("Calculating reconstruction quality...")
    total_similarity = 0
    sample_size = min(50, len(sequence) - 1)

    step = len(sequence) // sample_size

    for i in range(0, len(sequence) - 1, step):
        if i >= len(sequence) - 1:
            break

        img1 = frames_data[sequence[i]]["image"]
        img2 = frames_data[sequence[i + 1]]["image"]

        similarity = calc.compute_ssim(img1, img2)
        total_similarity += similarity

    avg_similarity = total_similarity / sample_size
    print(f"  Average consecutive frame similarity: {avg_similarity:.2%}")

    return avg_similarity


def save_sequence(sequence, output_file="output/sequence.txt"):
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        for idx in sequence:
            f.write(f"{idx}\n")

    print(f"  Sequence saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct jumbled video frames")
    parser.add_argument("video_path", type=str, help="Path to jumbled video file")
    parser.add_argument(
        "--output",
        type=str,
        default="output/reconstructed.mp4",
        help="Output video path (default: output/reconstructed.mp4)",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Output video FPS (default: 30)"
    )

    args = parser.parse_args()

    main(args.video_path, args.output, args.fps)
