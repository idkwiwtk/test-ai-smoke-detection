"""Download sample smoke/fire images for testing.

Uses the D-Fire dataset from GitHub (CC0 license).
For a quick start, downloads a small subset of test images.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from config import DATA_DIR, TEST_IMAGES_DIR


def download_dfire_sample():
    """Download D-Fire dataset sample images via git sparse checkout."""
    repo_url = "https://github.com/gaiasd/DFireDataset.git"
    clone_dir = DATA_DIR / "raw" / "DFireDataset"

    if clone_dir.exists():
        print(f"D-Fire dataset already exists at {clone_dir}")
        return clone_dir

    print("Downloading D-Fire dataset (this may take a while)...")
    print(f"Repository: {repo_url}")

    clone_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
            check=True,
        )
        print(f"Downloaded to {clone_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        print("Please manually download from: https://github.com/gaiasd/DFireDataset")
        sys.exit(1)

    return clone_dir


def organize_test_images(source_dir: Path, max_per_category: int = 50):
    """Organize downloaded images into test directory structure."""
    # D-Fire has test/images and test/labels directories
    test_img_dir = source_dir / "test" / "images"

    if not test_img_dir.exists():
        # Try alternative structure
        for candidate in [source_dir / "images" / "test", source_dir / "test"]:
            if candidate.exists():
                test_img_dir = candidate
                break

    if not test_img_dir.exists():
        print(f"Could not find test images in {source_dir}")
        print("Available directories:")
        for p in source_dir.rglob("*"):
            if p.is_dir():
                print(f"  {p.relative_to(source_dir)}")
        return

    test_label_dir = test_img_dir.parent / "labels"

    # Create output directories
    smoke_dir = TEST_IMAGES_DIR / "smoke"
    fire_dir = TEST_IMAGES_DIR / "fire"
    normal_dir = TEST_IMAGES_DIR / "normal"

    for d in [smoke_dir, fire_dir, normal_dir]:
        d.mkdir(parents=True, exist_ok=True)

    smoke_count = fire_count = normal_count = 0
    images = sorted(test_img_dir.glob("*"))

    for img_path in images:
        if not img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            continue

        label_path = test_label_dir / f"{img_path.stem}.txt"
        classes_in_image = set()

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        classes_in_image.add(int(parts[0]))

        # D-Fire: 0=fire, 1=smoke (check the actual mapping)
        has_fire = 0 in classes_in_image
        has_smoke = 1 in classes_in_image

        if has_smoke and smoke_count < max_per_category:
            shutil.copy2(img_path, smoke_dir / img_path.name)
            smoke_count += 1
        elif has_fire and fire_count < max_per_category:
            shutil.copy2(img_path, fire_dir / img_path.name)
            fire_count += 1
        elif not classes_in_image and normal_count < max_per_category:
            shutil.copy2(img_path, normal_dir / img_path.name)
            normal_count += 1

    print(f"\nOrganized test images:")
    print(f"  Smoke:  {smoke_count}")
    print(f"  Fire:   {fire_count}")
    print(f"  Normal: {normal_count}")
    print(f"  Total:  {smoke_count + fire_count + normal_count}")


def main():
    parser = argparse.ArgumentParser(description="Download smoke/fire test dataset")
    parser.add_argument("--max-per-category", type=int, default=50, help="Max images per category")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, only organize")
    parser.add_argument("--source", type=str, default=None, help="Custom source directory")
    args = parser.parse_args()

    if args.source:
        source_dir = Path(args.source)
    elif not args.skip_download:
        source_dir = download_dfire_sample()
    else:
        source_dir = DATA_DIR / "raw" / "DFireDataset"

    if source_dir.exists():
        organize_test_images(source_dir, args.max_per_category)
    else:
        print(f"Source directory not found: {source_dir}")


if __name__ == "__main__":
    main()
