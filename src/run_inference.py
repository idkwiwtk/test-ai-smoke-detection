"""Smoke/Fire detection inference pipeline."""

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

from config import (
    CONFIDENCE_THRESHOLD,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    IOU_THRESHOLD,
    MODEL_WEIGHTS,
    PREDICTIONS_DIR,
    RESULTS_DIR,
    TEST_IMAGES_DIR,
)


def load_model(weights_path: Path) -> YOLO:
    model = YOLO(str(weights_path))
    print(f"Model loaded: {weights_path.name}")
    print(f"Classes: {model.names}")
    return model


def run_single(model: YOLO, image_path: Path, conf: float = CONFIDENCE_THRESHOLD) -> dict:
    start = time.time()
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )
    elapsed = time.time() - start

    result = results[0]
    detections = []
    for i in range(len(result.boxes)):
        boxes = result.boxes
        detections.append({
            "class_id": int(boxes.cls[i]),
            "class_name": result.names[int(boxes.cls[i])],
            "confidence": round(float(boxes.conf[i]), 4),
            "bbox_xyxy": [round(c, 1) for c in boxes.xyxy[i].tolist()],
        })

    return {
        "image": str(image_path),
        "image_name": image_path.name,
        "detections": detections,
        "num_detections": len(detections),
        "inference_time_ms": round(elapsed * 1000, 1),
    }


def run_batch(model: YOLO, image_dir: Path, conf: float = CONFIDENCE_THRESHOLD) -> list[dict]:
    image_paths = sorted(
        p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"No images found in {image_dir}")
        return []

    print(f"Running inference on {len(image_paths)} images (conf={conf})...")
    all_results = []
    for i, img_path in enumerate(image_paths):
        result = run_single(model, img_path, conf=conf)
        n = result["num_detections"]
        status = f"  [{i+1}/{len(image_paths)}] {img_path.name}: {n} detection(s)"
        if n > 0:
            classes = [d["class_name"] for d in result["detections"]]
            status += f" ({', '.join(classes)})"
        print(status)
        all_results.append(result)

    return all_results


def save_results(results: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Smoke/Fire Detection Inference")
    parser.add_argument("--weights", type=str, default=str(MODEL_WEIGHTS), help="Model weights path")
    parser.add_argument("--source", type=str, default=str(TEST_IMAGES_DIR), help="Image file or directory")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "predictions.json"), help="Output JSON path")
    args = parser.parse_args()

    model = load_model(Path(args.weights))
    source = Path(args.source)

    if source.is_file():
        result = run_single(model, source, conf=args.conf)
        results = [result]
        print(f"\nDetections: {result['num_detections']}")
        for d in result["detections"]:
            print(f"  - {d['class_name']}: {d['confidence']:.2%}")
    else:
        results = run_batch(model, source, conf=args.conf)

    if results:
        save_results(results, Path(args.output))

    total = len(results)
    detected = sum(1 for r in results if r["num_detections"] > 0)
    print(f"\nSummary: {detected}/{total} images with detections")


if __name__ == "__main__":
    main()
