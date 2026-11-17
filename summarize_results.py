"""Create a comparison CSV and chart from completed model metrics JSON files."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path("outputs")


def main():
    records = []
    for path in sorted(OUTPUT_DIR.glob("*_metrics.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            records.append({
                "model": data.get("model", path.stem.replace("_metrics", "")),
                "test_accuracy": data.get("accuracy"),
                "weighted_precision": data.get("weighted_precision"),
                "weighted_recall": data.get("weighted_recall"),
                "weighted_f1_score": data.get("weighted_f1_score"),
                "best_validation_accuracy": data.get("best_validation_accuracy"),
                "test_samples": data.get("test_samples"),
                "training_time_seconds": data.get("training_time_seconds"),
            })
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping {path}: {exc}")

    if not records:
        raise SystemExit("No *_metrics.json files found in outputs/. Train at least one model first.")

    frame = pd.DataFrame(records).sort_values("test_accuracy", ascending=False, na_position="last")
    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / "model_comparison.csv"
    frame.to_csv(csv_path, index=False)

    metric_columns = ["test_accuracy", "weighted_precision", "weighted_recall", "weighted_f1_score"]
    plot_frame = frame.set_index("model")[metric_columns].dropna(how="all")
    ax = plot_frame.plot(kind="bar", figsize=(11, 6), ylim=(0, 1), rot=20)
    ax.set_title("OCT model comparison on the test split")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    chart_path = OUTPUT_DIR / "model_comparison.png"
    plt.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(frame.to_string(index=False))
    print(f"\nSaved comparison table: {csv_path}")
    print(f"Saved comparison chart: {chart_path}")
    print("Only models with real metrics files are included; no results are fabricated.")


if __name__ == "__main__":
    main()
