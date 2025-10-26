import cv2
import numpy as np
from pathlib import Path
from similarity_metrics import SimilarityCalculator
from tqdm import tqdm
import json


class ReconstructionValidator:
    def __init__(self, frames_dir="frames"):
        self.frames_dir = Path(frames_dir)
        self.calc = SimilarityCalculator()

    def load_sequence_from_file(self, sequence_file):
        with open(sequence_file, "r") as f:
            sequence = [int(line.strip()) for line in f if line.strip()]
        return sequence

    def calculate_framewise_similarity(self, sequence):
        print("\nCalculating frame-wise similarity scores...")

        similarities = []

        for i in tqdm(range(len(sequence) - 1), desc="Computing SSIM"):
            frame1_path = self.frames_dir / f"frame_{sequence[i]:04d}.jpg"
            frame2_path = self.frames_dir / f"frame_{sequence[i + 1]:04d}.jpg"

            img1 = cv2.imread(str(frame1_path))
            img2 = cv2.imread(str(frame2_path))

            if img1 is None or img2 is None:
                print(
                    f"Warning: Could not load frames {sequence[i]} or {sequence[i + 1]}"
                )
                continue

            ssim_score = self.calc.compute_ssim(img1, img2)
            similarities.append(ssim_score)

        return similarities

    def generate_quality_report(
        self, sequence, output_file="output/quality_report.txt"
    ):
        print("\nGenerating quality report...")

        similarities = self.calculate_framewise_similarity(sequence)

        if not similarities:
            print("Error: No similarities calculated")
            return

        avg_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)
        max_similarity = np.max(similarities)
        std_similarity = np.std(similarities)

        threshold = avg_similarity - std_similarity
        problem_frames = [i for i, s in enumerate(similarities) if s < threshold]

        report = []
        report.append("=" * 70)
        report.append("RECONSTRUCTION QUALITY REPORT")
        report.append("=" * 70)
        report.append("")
        report.append(f"Total Frames: {len(sequence)}")
        report.append(f"Total Transitions: {len(similarities)}")
        report.append("")
        report.append("SIMILARITY STATISTICS")
        report.append("-" * 70)
        report.append(
            f"  Average Similarity:     {avg_similarity:.4f} ({avg_similarity * 100:.2f}%)"
        )
        report.append(
            f"  Minimum Similarity:     {min_similarity:.4f} ({min_similarity * 100:.2f}%)"
        )
        report.append(
            f"  Maximum Similarity:     {max_similarity:.4f} ({max_similarity * 100:.2f}%)"
        )
        report.append(f"  Standard Deviation:     {std_similarity:.4f}")
        report.append("")
        report.append("QUALITY ASSESSMENT")
        report.append("-" * 70)

        if avg_similarity >= 0.95:
            quality = "EXCELLENT"
            assessment = "Reconstruction is highly accurate with minimal errors."
        elif avg_similarity >= 0.90:
            quality = "GOOD"
            assessment = "Reconstruction is accurate with minor imperfections."
        elif avg_similarity >= 0.85:
            quality = "ACCEPTABLE"
            assessment = "Reconstruction is reasonable but may have some errors."
        else:
            quality = "NEEDS IMPROVEMENT"
            assessment = "Reconstruction may have significant errors."

        report.append(f"  Overall Quality:        {quality}")
        report.append(f"  Assessment:             {assessment}")
        report.append("")

        if problem_frames:
            report.append("POTENTIAL PROBLEM AREAS")
            report.append("-" * 70)
            report.append(f"  Found {len(problem_frames)} transitions below threshold")
            report.append(f"  Threshold: {threshold:.4f}")
            report.append("")
            report.append("  Frame Index | Similarity | Frames")
            report.append("  " + "-" * 40)

            for idx in problem_frames[:10]:
                report.append(
                    f"  {idx:11d} | {similarities[idx]:10.4f} | {sequence[idx]:04d} -> {sequence[idx + 1]:04d}"
                )

            if len(problem_frames) > 10:
                report.append(f"  ... and {len(problem_frames) - 10} more")
        else:
            report.append("No significant problem areas detected.")

        report.append("")
        report.append("SIMILARITY DISTRIBUTION")
        report.append("-" * 70)

        bins = [0.0, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
        hist, _ = np.histogram(similarities, bins=bins)

        for i in range(len(bins) - 1):
            count = hist[i]
            percentage = (count / len(similarities)) * 100
            bar = "█" * int(percentage / 2)
            report.append(
                f"  {bins[i]:.2f}-{bins[i + 1]:.2f}: {bar} {count:3d} ({percentage:5.1f}%)"
            )

        report.append("")
        report.append("=" * 70)

        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w") as f:
            f.write("\n".join(report))

        print("\n" + "\n".join(report))

        json_data = {
            "total_frames": len(sequence),
            "total_transitions": len(similarities),
            "average_similarity": float(avg_similarity),
            "min_similarity": float(min_similarity),
            "max_similarity": float(max_similarity),
            "std_similarity": float(std_similarity),
            "quality_rating": quality,
            "problem_frame_count": len(problem_frames),
            "similarities": [float(s) for s in similarities],
        }

        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"\nReports saved:")
        print(f"  Text: {output_path}")
        print(f"  JSON: {json_path}")

        return avg_similarity

    def create_similarity_heatmap(
        self, similarities, output_file="output/similarity_heatmap.jpg"
    ):
        print("\nCreating similarity heatmap...")

        width = 1920
        height = 400

        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        points_per_pixel = len(similarities) / width

        for x in range(width):
            start_idx = int(x * points_per_pixel)
            end_idx = int((x + 1) * points_per_pixel)

            if end_idx > len(similarities):
                end_idx = len(similarities)

            if start_idx >= len(similarities):
                break

            avg_sim = np.mean(similarities[start_idx:end_idx])

            y = int(height - (avg_sim * height))

            if avg_sim > 0.95:
                color = (0, 255, 0)
            elif avg_sim > 0.90:
                color = (0, 255, 255)
            elif avg_sim > 0.85:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)

            cv2.line(canvas, (x, height), (x, y), color, 1)

        cv2.putText(
            canvas,
            "Frame-wise Similarity Scores",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            f"Avg: {np.mean(similarities):.4f}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.line(
            canvas, (0, int(height * 0.05)), (width, int(height * 0.05)), (0, 255, 0), 1
        )
        cv2.line(
            canvas,
            (0, int(height * 0.10)),
            (width, int(height * 0.10)),
            (0, 255, 255),
            1,
        )

        cv2.imwrite(output_file, canvas)
        print(f"  Saved to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate reconstruction quality")
    parser.add_argument(
        "--sequence",
        type=str,
        default="output/sequence.txt",
        help="Path to sequence file",
    )

    args = parser.parse_args()

    validator = ReconstructionValidator()

    print(f"Loading sequence from: {args.sequence}")
    sequence = validator.load_sequence_from_file(args.sequence)
    print(f"  Loaded {len(sequence)} frames")

    avg_sim = validator.generate_quality_report(sequence)

    similarities = validator.calculate_framewise_similarity(sequence)
    validator.create_similarity_heatmap(similarities)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Average Similarity: {avg_sim:.2%}")


if __name__ == "__main__":
    main()
