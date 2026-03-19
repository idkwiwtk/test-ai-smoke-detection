"""Evaluation pipeline for smoke/fire detection model."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)

from config import METRICS_DIR, PLOTS_DIR, RESULTS_DIR, TARGET_CLASSES
from visualize import plot_confusion_matrix, plot_precision_recall, plot_threshold_sweep


def build_ground_truth(test_images_dir: Path) -> dict[str, str]:
    """Build ground truth labels from directory structure.

    Expected structure:
        test_images/smoke/     -> label "smoke"
        test_images/fire/      -> label "fire"
        test_images/normal/    -> label "normal"
        test_images/edge_cases/ -> label "normal"
    """
    gt = {}
    label_map = {
        "smoke": "smoke",
        "fire": "fire",
        "normal": "normal",
        "edge_cases": "normal",
    }
    for folder_name, label in label_map.items():
        folder = test_images_dir / folder_name
        if not folder.exists():
            continue
        for img in folder.iterdir():
            if img.is_file():
                gt[img.name] = label
    return gt


def image_level_eval(predictions: list[dict], ground_truth: dict[str, str], threshold: float = 0.25):
    """Evaluate at image level: did the model detect the correct class?"""
    y_true_smoke, y_pred_smoke, y_scores_smoke = [], [], []
    y_true_fire, y_pred_fire, y_scores_fire = [], [], []

    for pred in predictions:
        fname = pred["image_name"]
        gt_label = ground_truth.get(fname)
        if gt_label is None:
            continue

        dets = pred["detections"]

        # Smoke evaluation
        has_smoke_gt = 1 if gt_label == "smoke" else 0
        smoke_dets = [d for d in dets if d["class_name"] == "smoke" and d["confidence"] >= threshold]
        max_smoke_conf = max((d["confidence"] for d in dets if d["class_name"] == "smoke"), default=0.0)
        y_true_smoke.append(has_smoke_gt)
        y_pred_smoke.append(1 if len(smoke_dets) > 0 else 0)
        y_scores_smoke.append(max_smoke_conf)

        # Fire evaluation
        has_fire_gt = 1 if gt_label == "fire" else 0
        fire_dets = [d for d in dets if d["class_name"] == "fire" and d["confidence"] >= threshold]
        max_fire_conf = max((d["confidence"] for d in dets if d["class_name"] == "fire"), default=0.0)
        y_true_fire.append(has_fire_gt)
        y_pred_fire.append(1 if len(fire_dets) > 0 else 0)
        y_scores_fire.append(max_fire_conf)

    results = {}
    for cls_name, y_true, y_pred, y_scores in [
        ("smoke", y_true_smoke, y_pred_smoke, y_scores_smoke),
        ("fire", y_true_fire, y_pred_fire, y_scores_fire),
    ]:
        if len(set(y_true)) < 2:
            continue
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        ap = average_precision_score(y_true, y_scores)
        report = classification_report(y_true, y_pred, target_names=[f"no_{cls_name}", cls_name])
        results[cls_name] = {
            "confusion_matrix": cm,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            "average_precision": round(ap, 4),
            "report": report,
            "y_true": y_true,
            "y_scores": y_scores,
        }

    return results


def threshold_sweep(predictions: list[dict], ground_truth: dict[str, str], class_name: str = "smoke") -> list[dict]:
    """Sweep confidence thresholds to find optimal operating point."""
    thresholds = np.arange(0.05, 0.96, 0.05)
    results = []

    for t in thresholds:
        y_true, y_pred = [], []
        for pred in predictions:
            fname = pred["image_name"]
            gt_label = ground_truth.get(fname)
            if gt_label is None:
                continue
            has_gt = 1 if gt_label == class_name else 0
            dets = [d for d in pred["detections"] if d["class_name"] == class_name and d["confidence"] >= t]
            y_true.append(has_gt)
            y_pred.append(1 if len(dets) > 0 else 0)

        if len(set(y_true)) < 2:
            continue
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        results.append({
            "threshold": round(float(t), 2),
            "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0,
            "recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0,
            "f1": round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) > 0 else 0,
            "fp_rate": round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })

    return results


def run_evaluation(predictions_path: Path, test_images_dir: Path, output_dir: Path):
    with open(predictions_path, encoding="utf-8") as f:
        predictions = json.load(f)

    gt = build_ground_truth(test_images_dir)
    if not gt:
        print("No ground truth found. Place images in smoke/, fire/, normal/, edge_cases/ subdirectories.")
        return

    print(f"Ground truth: {len(gt)} images")
    label_counts = {}
    for label in gt.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    # Image-level evaluation
    eval_results = image_level_eval(predictions, gt)

    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for cls_name, cls_results in eval_results.items():
        print(f"\n{'='*50}")
        print(f"  {cls_name.upper()} Detection Results")
        print(f"{'='*50}")
        print(cls_results["report"])
        print(f"Average Precision: {cls_results['average_precision']:.4f}")

        # Confusion matrix plot
        plot_confusion_matrix(
            cls_results["confusion_matrix"],
            [f"No {cls_name}", cls_name.capitalize()],
            plots_dir / f"confusion_matrix_{cls_name}.png",
        )

        # PR curve
        if cls_results["y_true"] and cls_results["y_scores"]:
            plot_precision_recall(
                cls_results["y_true"],
                cls_results["y_scores"],
                plots_dir / f"pr_curve_{cls_name}.png",
                cls_name,
            )

        # Threshold sweep
        sweep = threshold_sweep(predictions, gt, cls_name)
        if sweep:
            plot_threshold_sweep(sweep, plots_dir / f"threshold_sweep_{cls_name}.png")

            # Save sweep as JSON
            with open(metrics_dir / f"threshold_sweep_{cls_name}.json", "w") as f:
                json.dump(sweep, f, indent=2)

            # Find best F1 threshold
            best = max(sweep, key=lambda x: x["f1"])
            print(f"\nBest threshold (F1): {best['threshold']}")
            print(f"  Precision: {best['precision']:.4f}")
            print(f"  Recall:    {best['recall']:.4f}")
            print(f"  F1:        {best['f1']:.4f}")

            # Find threshold for 90%+ recall
            high_recall = [r for r in sweep if r["recall"] >= 0.9]
            if high_recall:
                best_hr = max(high_recall, key=lambda x: x["precision"])
                print(f"\nBest threshold (Recall>=90%): {best_hr['threshold']}")
                print(f"  Precision: {best_hr['precision']:.4f}")
                print(f"  Recall:    {best_hr['recall']:.4f}")
                print(f"  FP rate:   {best_hr['fp_rate']:.4f}")

    # Save summary
    summary = {
        "total_images": len(predictions),
        "ground_truth_images": len(gt),
    }
    for cls_name, cls_results in eval_results.items():
        summary[cls_name] = {
            "tp": cls_results["tp"], "fp": cls_results["fp"],
            "fn": cls_results["fn"], "tn": cls_results["tn"],
            "precision": round(cls_results["precision"], 4),
            "recall": round(cls_results["recall"], 4),
            "f1": round(cls_results["f1"], 4),
            "average_precision": cls_results["average_precision"],
        }

    with open(metrics_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {metrics_dir / 'evaluation_summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate smoke/fire detection model")
    parser.add_argument("--predictions", type=str, default=str(RESULTS_DIR / "predictions.json"))
    parser.add_argument("--test-images", type=str, required=True, help="Test images directory")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    run_evaluation(Path(args.predictions), Path(args.test_images), Path(args.output))


if __name__ == "__main__":
    main()
