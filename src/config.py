from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEST_IMAGES_DIR = DATA_DIR / "test_images"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
MODELS_DIR = PROJECT_ROOT / "models" / "weights"
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

# Model settings
MODEL_WEIGHTS = MODELS_DIR / "fire_smoke_best.pt"
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640

# Target classes
TARGET_CLASSES = ["fire", "smoke"]

# Image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
