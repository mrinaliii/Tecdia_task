import numpy as np
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
                distances = self.similarity_matrix[current_idx].copy()
                distances[list(used)] = np.inf

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

        for i in tqdm(range(len(sequence) - window_size), desc="Refining"):
            window = refined[i : i + window_size]

            if window_size <= 4:
                best_window, improved = self._optimize_window(window)
                if improved:
                    refined[i : i + window_size] = best_window
                    improvements += 1

        elapsed = time.time() - start_time
        print(f"✓ Refinement complete: {improvements} improvements in {elapsed:.2f}s")

        return refined

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


if __name__ == "__main__":
    from frame_loader import FrameLoader

    loader = FrameLoader()
    frames_data = loader.load_all_frames_parallel()
    similarity_matrix, indices = loader.compute_hash_similarity_matrix(frames_data)

    # Reconstruct
    reconstructor = FrameReconstructor(frames_data, similarity_matrix, indices)
    sequence = reconstructor.greedy_nearest_neighbor()

    print(f"\nReconstructed sequence (first 10): {sequence[:10]}")
