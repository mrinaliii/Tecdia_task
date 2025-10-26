import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm
from similarity_metrics import SimilarityCalculator
import time


class FrameReconstructor:
    def __init__(self, frames_data, similarity_matrix, frame_indices):
        self.frames_data = frames_data
        self.similarity_matrix = similarity_matrix
        self.frame_indices = frame_indices
        self.calc = SimilarityCalculator()
        self.n_frames = len(frame_indices)

    def find_best_starting_frame(self):
        print("Finding optimal starting frame...")

        avg_distances = np.mean(self.similarity_matrix, axis=1)

        candidates = np.argsort(avg_distances)[-10:]

        print(f"  Top starting frame candidates: {candidates[:5]}")

        return candidates[-1]

    def greedy_nearest_neighbor(self, start_idx=None):
        print("\nConstructing frame sequence...")
        start_time = time.time()

        if start_idx is None:
            start_idx = self.find_best_starting_frame()

        current_idx = start_idx
        sequence = [self.frame_indices[current_idx]]
        used = {current_idx}

        with tqdm(total=self.n_frames - 1, desc="Building sequence") as pbar:
            while len(sequence) < self.n_frames:
                distances = (
                    self.similarity_matrix[current_idx].astype(np.float32).copy()
                )

                for used_idx in used:
                    distances[used_idx] = np.finfo(np.float32).max

                next_idx = np.argmin(distances)

                sequence.append(self.frame_indices[next_idx])
                used.add(next_idx)
                current_idx = next_idx

                pbar.update(1)

        elapsed = time.time() - start_time
        print(f"✓ Sequence constructed in {elapsed:.2f} seconds")

        return sequence

    def refine_with_ssim(self, sequence, window_size=5):
        print(f"\nRefining sequence with SSIM (window size={window_size})...")
        start_time = time.time()

        refined = sequence.copy()
        improvements = 0
        problem_areas = self._identify_problem_areas(sequence)

        print(f"  Identified {len(problem_areas)} potential problem areas")
        print(
            f"  Refining {len(problem_areas)} windows instead of {len(sequence) - window_size}"
        )
        for i in tqdm(problem_areas, desc="Refining problem areas"):
            if i + window_size > len(sequence):
                continue

            window = refined[i : i + window_size]

            if window_size <= 4:
                best_window, improved = self._optimize_window(window)
                if improved:
                    refined[i : i + window_size] = best_window
                    improvements += 1

        elapsed = time.time() - start_time
        print(f"✓ Refinement complete: {improvements} improvements in {elapsed:.2f}s")

        return refined

    def _identify_problem_areas(self, sequence, threshold_percentile=75):
        distances = []
        for i in range(len(sequence) - 1):
            idx1 = self.frame_indices.index(sequence[i])
            idx2 = self.frame_indices.index(sequence[i + 1])
            distance = self.similarity_matrix[idx1, idx2]
            distances.append(distance)

        threshold = np.percentile(distances, threshold_percentile)

        problem_indices = []
        for i, dist in enumerate(distances):
            if dist > threshold:
                for offset in range(-1, 2):  # Check i-1, i, i+1
                    idx = i + offset
                    if 0 <= idx < len(sequence) - 4:  # Ensure window fits
                        if idx not in problem_indices:
                            problem_indices.append(idx)

        return sorted(problem_indices)

    def _optimize_window(self, window):
        from itertools import permutations

        best_score = self._score_window(window)
        best_window = window

        for perm in permutations(window):
            score = self._score_window(list(perm))
            if score > best_score:
                best_score = score
                best_window = list(perm)

        return best_window, (best_window != window)

    def _score_window(self, window):
        total_score = 0

        for i in range(len(window) - 1):
            img1 = self.frames_data[window[i]]["image"]
            img2 = self.frames_data[window[i + 1]]["image"]

            score = self.calc.compute_ssim(img1, img2)
            total_score += score

        return total_score / (len(window) - 1) if len(window) > 1 else 0

    def try_reverse_sequence(self, sequence):
        print("\nTesting sequence direction...")

        sample_size = min(10, len(sequence) - 1)

        forward_score = self._score_window(sequence[: sample_size + 1])
        reverse_score = self._score_window(sequence[: sample_size + 1][::-1])

        if reverse_score > forward_score:
            print(
                f"  Reverse sequence is better ({reverse_score:.4f} vs {forward_score:.4f})"
            )
            return sequence[::-1]
        else:
            print(
                f"  Forward sequence is better ({forward_score:.4f} vs {reverse_score:.4f})"
            )
            return sequence

    def _optimize_window_parallel(self, windows):
        results = {}

        max_workers = min(4, os.cpu_count() or 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for idx, window in windows:
                future = executor.submit(self._optimize_window, window)
                future_to_idx[future] = idx

            for future in future_to_idx:
                idx = future_to_idx[future]
                results[idx] = future.result()

        return results

    def refine_with_ssim_parallel(self, sequence, window_size=4):
        print(f"\nRefining sequence with SSIM (parallel, window size={window_size})...")
        start_time = time.time()

        refined = sequence.copy()

        problem_areas = self._identify_problem_areas(sequence)
        print(f"  Identified {len(problem_areas)} potential problem areas")

        if len(problem_areas) == 0:
            print("  No problem areas found - sequence looks good!")
            return refined

        windows = []
        for i in problem_areas:
            if i + window_size <= len(sequence):
                window = refined[i : i + window_size]
                windows.append((i, window))

        print(f"  Processing {len(windows)} windows in parallel...")

        results = self._optimize_window_parallel(windows)

        improvements = 0
        for i, (best_window, improved) in results.items():
            if improved:
                refined[i : i + window_size] = best_window
                improvements += 1

        elapsed = time.time() - start_time
        print(f"✓ Refinement complete: {improvements} improvements in {elapsed:.2f}s")

        return refined


if __name__ == "__main__":
    from frame_loader import FrameLoader

    loader = FrameLoader()
    frames_data = loader.load_all_frames_parallel()
    similarity_matrix, indices = loader.compute_hash_similarity_matrix(frames_data)

    reconstructor = FrameReconstructor(frames_data, similarity_matrix, indices)
    sequence = reconstructor.greedy_nearest_neighbor()

    print(f"\nReconstructed sequence (first 10): {sequence[:10]}")
