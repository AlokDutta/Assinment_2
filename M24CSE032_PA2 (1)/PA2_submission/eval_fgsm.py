"""Standalone FGSM evaluation using saved LID weights.

Re-runs the FGSM attack with early-stopping + reduced step size to find the
minimum perturbation that flips LID from Hindi to English while keeping
SNR > 40 dB.  Updates outputs/pipeline_results.json with the fgsm key.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_fgsm.py
"""

import os
import sys
import json
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.utils import load_audio, get_device
from src.part1_stt.lid import FrameLevelLID
from src.part4_adversarial.fgsm_attack import FGSMAttacker


def main():
    device = get_device()
    audio_path = str(BASE_DIR / "denoised_segment.wav")
    weights_path = str(BASE_DIR / "outputs" / "lid_weights.pt")
    output_dir = str(BASE_DIR / "outputs")

    print("=" * 60)
    print("FGSM EVALUATION (inference-only, using saved LID weights)")
    print(f"Device: {device}")
    print("=" * 60)

    if not Path(weights_path).exists():
        print(f"ERROR: {weights_path} not found. Run the pipeline or eval_lid first.")
        return

    lid = FrameLevelLID()
    lid.load_weights(weights_path)

    waveform, sr = load_audio(audio_path)
    frame_results = lid.predict(waveform, sr)
    language_segments = lid.get_language_segments(frame_results)

    hindi_start = 0.0
    for seg in language_segments:
        if seg["lang"] == "hi" and (seg["end"] - seg["start"]) >= 5.0:
            hindi_start = seg["start"]
            break
    print(f"[FGSM] Chosen Hindi segment start: {hindi_start:.2f}s")

    attacker = FGSMAttacker(lid)
    fgsm_result = attacker.run_attack(audio_path, hindi_start,
                                      duration_sec=5.0, output_dir=output_dir)

    pr_path = str(Path(output_dir) / "pipeline_results.json")
    if Path(pr_path).exists():
        with open(pr_path, "r") as f:
            pipeline_results = json.load(f)
    else:
        pipeline_results = {}

    pipeline_results["fgsm"] = {
        "epsilon": fgsm_result.get("epsilon"),
        "snr": fgsm_result.get("snr"),
        "flipped": fgsm_result.get("flipped"),
    }

    with open(pr_path, "w") as f:
        json.dump(pipeline_results, f, indent=2)

    snr_val = fgsm_result.get("snr")
    eps_val = fgsm_result.get("epsilon")
    print("\n--- FGSM Results ---")
    if eps_val is not None:
        print(f"  Epsilon: {eps_val:.6f}")
    else:
        print("  Epsilon: N/A")
    if snr_val is not None and snr_val != float("inf"):
        print(f"  SNR:     {snr_val:.2f} dB (threshold > 40 dB)")
    else:
        print(f"  SNR:     {snr_val} dB (threshold > 40 dB)")
    print(f"  Flipped: {fgsm_result.get('flipped')}")
    print(f"  Status:  {'PASS' if snr_val and snr_val > 40 and fgsm_result.get('flipped') else 'FAIL'}")
    print(f"\n  Updated {pr_path}")


if __name__ == "__main__":
    main()
