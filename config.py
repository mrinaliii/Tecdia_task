class Config:
    PARALLEL_WORKERS = 10
    USE_PARALLEL = True

    SSIM_WORKERS = 4

    HASH_SIZE = 16
    SSIM_RESOLUTION = (480, 270)

    REFINEMENT_WINDOW_SIZE = 4
    ENABLE_SSIM_REFINEMENT = True
    STARTING_FRAME_CANDIDATES = 10

    REFINE_PROBLEM_AREAS_ONLY = True
    PROBLEM_THRESHOLD_PERCENTILE = 75

    @classmethod
    def quick_mode(cls):
        cls.PARALLEL_WORKERS = 10
        cls.HASH_SIZE = 12
        cls.SSIM_RESOLUTION = (320, 180)
        cls.REFINEMENT_WINDOW_SIZE = 3
        cls.ENABLE_SSIM_REFINEMENT = False
        cls.QUALITY_SAMPLE_SIZE = 20
        print("Quick mode enabled (fast, lower quality)")

    @classmethod
    def fast_mode(cls):
        cls.PARALLEL_WORKERS = 10
        cls.HASH_SIZE = 16
        cls.SSIM_RESOLUTION = (480, 270)
        cls.REFINEMENT_WINDOW_SIZE = 4
        cls.ENABLE_SSIM_REFINEMENT = True
        cls.REFINE_PROBLEM_AREAS_ONLY = True
        cls.PROBLEM_THRESHOLD_PERCENTILE = 75
        cls.SSIM_WORKERS = 4
        cls.QUALITY_SAMPLE_SIZE = 50
        print("Fast mode enabled (good quality, 5x faster refinement)")

    @classmethod
    def balanced_mode(cls):
        cls.PARALLEL_WORKERS = 10
        cls.HASH_SIZE = 16
        cls.SSIM_RESOLUTION = (480, 270)
        cls.REFINEMENT_WINDOW_SIZE = 4
        cls.ENABLE_SSIM_REFINEMENT = True
        cls.REFINE_PROBLEM_AREAS_ONLY = True
        cls.SSIM_WORKERS = 4
        cls.QUALITY_SAMPLE_SIZE = 50
        print("Balanced mode enabled (default settings)")
