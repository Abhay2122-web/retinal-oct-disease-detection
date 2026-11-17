"""Train all five project architectures using the enhanced training script.

Run examples:
  python compare_models.py --epochs 5 --batch-size 16
  python compare_models.py --epochs 1 --max-train-samples 1000 --batch-size 8

The second command is a pipeline smoke test, not final evaluation.
"""
import argparse
import subprocess
import sys
from pathlib import Path

MODELS = ["custom_cnn", "resnet18", "vgg16", "mobilenet_v2", "densenet121"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    script = Path(__file__).with_name("train.py")
    if not script.exists():
        raise FileNotFoundError("train.py must be in the same folder.")

    for model_name in MODELS:
        command = [sys.executable, str(script), "--model", model_name,
                   "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                   "--max-train-samples", str(args.max_train_samples)]
        if args.freeze_backbone:
            command.append("--freeze-backbone")
        if args.no_pretrained:
            command.append("--no-pretrained")
        print("\n" + "=" * 72, flush=True)
        print(f"Starting model: {model_name}", flush=True)
        print("Command:", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=Path(__file__).parent)

    print("\nAll five model runs completed. Run summarize_results.py next.")


if __name__ == "__main__":
    main()
