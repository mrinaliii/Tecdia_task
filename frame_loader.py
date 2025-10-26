import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from similarity_metrics import SimilarityCalculator


class FrameLoader:
    def __init__(self, frames_dir="frames"):
        self.frames_dir = Path(frames_dir)
        self.calc = SimilarityCalculator()

    def get_frame_paths(self):
        frame_paths = sorted(self.frames_dir.glob("frame_*.jpg"))
        return frame_paths

    def load_frame_with_hash(self, frame_path):
        frame_idx = int(frame_path.stem.split("_")[1])

        image = cv2.imread(str(frame_path))

        if image is None:
            raise ValueError(f"Could not load frame: {frame_path}")

        phash = self.calc.perceptual_hash(image)

        return frame_idx, image, phash

    def load_all_frames_parallel(self, max_workers=None):
        frame_paths = self.get_frame_paths()
        frames_data = {}

        print(f"Loading {len(frame_paths)} frames in parallel...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.load_frame_with_hash, path): path
                for path in frame_paths
            }

            with tqdm(total=len(frame_paths), desc="Loading frames") as pbar:
                for future in as_completed(future_to_path):
                    try:
                        frame_idx, image, phash = future.result()
                        frames_data[frame_idx] = {
                            "image": image,
                            "hash": phash,
                            "index": frame_idx,
                        }
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error loading frame: {e}")

        print(f"✓ Loaded {len(frames_data)} frames")
        return frames_data

    def compute_hash_similarity_matrix(self, frames_data):
        n_frames = len(frames_data)
        indices = sorted(frames_data.keys())

        print(f"Computing hash similarity matrix for {n_frames} frames...")

        similarity_matrix = np.zeros((n_frames, n_frames), dtype=np.float32)

        with tqdm(
            total=n_frames * (n_frames - 1) // 2, desc="Computing similarities"
        ) as pbar:
            for i in range(n_frames):
                for j in range(i + 1, n_frames):
                    hash1 = frames_data[indices[i]]["hash"]
                    hash2 = frames_data[indices[j]]["hash"]
                    distance = self.calc.hamming_distance(hash1, hash2)

                    similarity_matrix[i, j] = distance
                    similarity_matrix[j, i] = distance
                    pbar.update(1)

        print(f"✓ Similarity matrix computed")
        return similarity_matrix, indices


if __name__ == "__main__":
    loader = FrameLoader()
    frames_data = loader.load_all_frames_parallel()

    print(f"\nLoaded frames: {len(frames_data)}")
    print(f"Sample frame data: {list(frames_data.keys())[:5]}")

    similarity_matrix, indices = loader.compute_hash_similarity_matrix(frames_data)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
