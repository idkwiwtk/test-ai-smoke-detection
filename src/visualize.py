"""Visualization utilities for smoke/fire detection results."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import CONFIDENCE_THRESHOLD, PLOTS_DIR, PREDICTIONS_DIR

# Colors: fire=red, smoke=orange, other=green
CLASS_COLORS = {
    "fire": (0, 0, 255),
    "smoke": (0, 165, 255),
}
DEFAULT_COLOR = (0, 255, 0)


def draw_detections(image_path: str, detections: list[dict], output_path: Path, conf_threshold: float = 0.0):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        return

    for det in detections:
        if det["confidence"] < conf_threshold:
            continue
        x1, y1, x2, y2 = [int(c) for c in det["bbox_xyxy"]]
        color = CLASS_COLORS.get(det["class_name"], DEFAULT_COLOR)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def draw_all_predictions(results_json: Path, output_dir: Path, conf_threshold: float = 0.0):
    with open(results_json, encoding="utf-8") as f:
        results = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for r in results:
        if r["num_detections"] == 0:
            continue
        out_path = output_dir / f"pred_{Path(r['image_name']).stem}.jpg"
        draw_detections(r["image"], r["detections"], out_path, conf_threshold)
        count += 1

    print(f"Saved {count} annotated images to {output_dir}")


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Detection Confusion Matrix")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")


def plot_precision_recall(y_true: list, y_scores: list, output_path: Path, class_name: str = "smoke"):
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, "b-", linewidth=2)
    ax.fill_between(recall, precision, alpha=0.1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve ({class_name})")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PR curve saved to {output_path}")


def plot_threshold_sweep(sweep_results: list[dict], output_path: Path):
    thresholds = [r["threshold"] for r in sweep_results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, [r["precision"] for r in sweep_results], "b-o", label="Precision", markersize=3)
    ax.plot(thresholds, [r["recall"] for r in sweep_results], "r-o", label="Recall", markersize=3)
    ax.plot(thresholds, [r["f1"] for r in sweep_results], "g-o", label="F1", markersize=3)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs Confidence Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Threshold sweep plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize detection results")
    parser.add_argument("--results", type=str, required=True, help="Path to predictions.json")
    parser.add_argument("--output", type=str, default=str(PREDICTIONS_DIR), help="Output directory")
    parser.add_argument("--conf", type=float, default=0.0, help="Min confidence to draw")
    args = parser.parse_args()

    draw_all_predictions(Path(args.results), Path(args.output), args.conf)


if __name__ == "__main__":
    main()
