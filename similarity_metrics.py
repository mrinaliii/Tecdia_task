import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor
import hashlib


class SimilarityCalculator:
    @staticmethod
    def perceptual_hash(image, hash_size=16):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        resized = cv2.resize(gray, (hash_size + 1, hash_size))

        diff = resized[:, 1:] > resized[:, :-1]
        return sum([2**i for (i, v) in enumerate(diff.flatten()) if v])

    @staticmethod
    def hamming_distance(hash1, hash2):
        return bin(hash1 ^ hash2).count("1")

    @staticmethod
    def compute_ssim(img1, img2):
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = img1, img2

        small_size = (480, 270)
        gray1 = cv2.resize(gray1, small_size)
        gray2 = cv2.resize(gray2, small_size)

        score, _ = ssim(gray1, gray2, full=True)
        return score

    @staticmethod
    def edge_similarity(img1, img2):
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = img1, img2

        small_size = (320, 180)
        gray1 = cv2.resize(gray1, small_size)
        gray2 = cv2.resize(gray2, small_size)

        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)

        correlation = np.corrcoef(edges1.flatten(), edges2.flatten())[0, 1]
        return max(0, correlation)


if __name__ == "__main__":
    calc = SimilarityCalculator()

    img1 = cv2.imread("frames/frame_0000.jpg")
    img2 = cv2.imread("frames/frame_0001.jpg")

    if img1 is not None and img2 is not None:
        hash1 = calc.perceptual_hash(img1)
        hash2 = calc.perceptual_hash(img2)
        ham_dist = calc.hamming_distance(hash1, hash2)

        print(f"Perceptual Hash Test:")
        print(f"  Hash 1: {hash1}")
        print(f"  Hash 2: {hash2}")
        print(f"  Hamming distance: {ham_dist} bits")

        ssim_score = calc.compute_ssim(img1, img2)
        print(f"\nSSIM Score: {ssim_score:.4f}")

        edge_score = calc.edge_similarity(img1, img2)
        print(f"Edge Similarity: {edge_score:.4f}")
