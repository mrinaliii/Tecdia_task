import time
import argparse
from pathlib import Path
import json

from frame_extractor import FrameExtractor
from frame_loader import FrameLoader
from frame_reconstructor import FrameReconstructor
from video_writer import VideoWriter
from similarity_metrics import SimilarityCalculator


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
        self.timings['total'] = total_time

        report = []
        report.append("=" * 60)
        report.append("JUMBLED FRAMES RECONSTRUCTION - EXECUTION LOG")
        report.append("=" * 60)
        report.append(f"\nTotal Execution Time: {total_time:.2f} seconds")
        report.append(f"                      ({total_time/60:.2f} minutes)")
        report.append("\nStep-by-Step Breakdown:")
        report.append("-" * 60)

        for step, duration in self.timings.items():
            if step != 'total':
                percentage = (duration / total_time) * 100
                report.append(f"  {step:.<45} {duration:>6.2f}s ({percentage:>5.1f}%)")

        report.append("=" * 60)

        with open(self.log_file, 'w') as f:
            f.write('\n'.join(report))

        print('\n' + '\n'.join(report))

        json_file = self.log_file.replace('.txt', '.json')
        with open(json_file, 'w') as f:
            json.dump(self.timings, f, indent=2)

        return total_time


def main(video_path, output_path="output/reconstructed.mp4", fps=30):
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
    print("-
