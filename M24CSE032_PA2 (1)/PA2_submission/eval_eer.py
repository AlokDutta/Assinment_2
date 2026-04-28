"""Standalone EER evaluation using 5-fold CV anti-spoofing.

Retrains the anti-spoofing classifier with k-fold CV and updates
outputs/pipeline_results.json with the mean EER.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_eer.py
"""

import os
import sys
import json
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.part4_adversarial.anti_spoof import run_anti_spoofing


def main():
    bonafide = str(BASE_DIR / "student_voice_ref.wav")
    spoof = str(BASE_DIR / "output_LRL_cloned.wav")
    output_dir = str(BASE_DIR / "outputs")

    print("=" * 60)
    print("ANTI-SPOOFING EER (5-fold cross-validation)")
    print("=" * 60)

    eer = run_anti_spoofing(bonafide, spoof, output_dir, n_folds=5)

    pr_path = BASE_DIR / "outputs" / "pipeline_results.json"
    if pr_path.exists():
        with open(pr_path) as f:
            pr = json.load(f)
    else:
        pr = {}

    pr["eer"] = round(eer, 4)
    with open(pr_path, "w") as f:
        json.dump(pr, f, indent=2)

    threshold = 0.10
    status = "✓ PASS" if eer < threshold else "✗ FAIL"
    print(f"\nEER: {eer*100:.2f}%  (threshold < 10%)  {status}")
    print(f"Updated {pr_path}")


if __name__ == "__main__":
    main()
