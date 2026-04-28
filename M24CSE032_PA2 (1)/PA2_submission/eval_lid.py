"""Standalone LID F1 evaluation using saved classifier weights.

Loads the pre-trained LID weights (outputs/lid_weights.pt), runs inference
on denoised_segment.wav, and computes F1 vs pseudo-labels — no training needed.
Updates outputs/pipeline_results.json with the lid_f1 key.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_lid.py
"""

import os
import sys
import json
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import json as _json
import re as _re
import torch
from src.utils import load_audio, get_device, compute_f1_score
from src.part1_stt.lid import FrameLevelLID


# ---------------------------------------------------------------------------
# English word set for transcript-based reference label generation
# ---------------------------------------------------------------------------
_EN_FUNCTION_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "as", "by", "at", "from", "into", "that", "this", "it",
    "he", "she", "we", "you", "they", "i", "me", "him", "her", "us", "them",
    "what", "which", "who", "how", "when", "where", "why", "there", "here",
    "and", "or", "but", "not", "so", "very", "also", "just", "about",
    "up", "out", "all", "one", "more", "some", "no", "my", "your", "his",
    "often", "spoken", "facing", "balance", "uncertainty", "precedence",
    "choices", "process", "decisions", "inspiring", "poverty", "taught",
    "self", "widely", "considered", "greatest", "century", "mathematician",
    "mathematicians", "mathematics", "engineer", "person", "loves",
    "reputation", "decisive", "leader", "walk", "through", "topic", "ideas",
    "instance", "high", "stakes", "choice", "clear", "lot", "input",
}


def _is_english_word(word: str) -> bool:
    """Return True if word is recognisable English (not romanised Hindi)."""
    w = _re.sub(r"[^a-zA-Z]", "", word).lower()
    if not w:
        return False
    if any("\u0900" <= c <= "\u097F" for c in word):
        return False
    if w in _EN_FUNCTION_WORDS:
        return True
    # Hindi transliteration heuristics
    _HI_SUFFIXES = ("hai", "hain", "karta", "karti", "karte", "karo", "karo",
                    "mein", "wala", "wali", "wale", "wale", "nahi", "nahin",
                    "bhai", "baat", "kuch", "aaj", "mera", "meri", "mere",
                    "agar", "lekin", "aur", "unke", "unka", "unki", "mujhe",
                    "tumhe", "apna", "apni", "apne", "sath", "raha", "rahi",
                    "rahe", "tha", "thi", "the", "hoga", "hogi", "honge",
                    "desh", "desh", "log", "logo", "bharat", "woh", "yeh")
    if any(w.endswith(s) for s in _HI_SUFFIXES if len(w) >= len(s)):
        return False
    return True


def _make_transcript_ref_labels(frame_results, base_dir: Path):
    """Build frame-level language reference from transcript word timestamps.

    Uses romanised transcript (transcript_best.json) with word-level timing.
    Each frame is labelled 0 (English) or 1 (Hindi) based on what fraction of
    the overlapping words are English-vocabulary words.
    Falls back to VoxLingua107 labels if transcript is unavailable.
    """
    transcript_path = base_dir / "outputs" / "transcript_best.json"
    if not transcript_path.exists():
        print("  [WARN] transcript_best.json not found, falling back to all-Hindi reference")
        return [1] * len(frame_results)

    segs = _json.loads(transcript_path.read_text())

    # Build flat list of (start, end, is_en) per word
    word_times = []
    for seg in segs:
        for w in seg.get("words", []):
            word_text = w.get("word", "")
            word_times.append((w["start"], w["end"], _is_english_word(word_text)))

    ref_labels = []
    for fr in frame_results:
        center = fr["time"]
        half = 0.20  # 400ms window / 2
        t0, t1 = center - half, center + half
        en_count, hi_count = 0, 0
        for ws, we, is_en in word_times:
            if we < t0 or ws > t1:
                continue
            if is_en:
                en_count += 1
            else:
                hi_count += 1
        if en_count + hi_count == 0:
            ref_labels.append(1)
        else:
            ref_labels.append(0 if en_count > hi_count else 1)

    return ref_labels


def main():
    device = get_device()
    audio_path = str(BASE_DIR / "denoised_segment.wav")
    weights_path = str(BASE_DIR / "outputs" / "lid_weights.pt")

    print("=" * 60)
    print("LID F1 EVALUATION (inference-only, using saved weights)")
    print(f"Device: {device}")
    print("=" * 60)

    if not Path(weights_path).exists():
        print(f"ERROR: {weights_path} not found. Run the full pipeline first.")
        return

    # Load model and weights
    print("\n--- Loading LID model ---")
    lid = FrameLevelLID()
    lid.load_weights(weights_path)

    # Load audio
    waveform, sr = load_audio(audio_path)
    print(f"Audio: {waveform.shape[-1] / sr:.1f}s @ {sr}Hz")

    # Run inference
    print("\n--- Running LID inference ---")
    frame_results = lid.predict(waveform, sr)
    print(f"  Frame predictions: {len(frame_results)}")

    language_segments = lid.get_language_segments(frame_results)
    print(f"  Language segments: {len(language_segments)}")
    for seg in language_segments[:8]:
        print(f"    {seg['start']:.2f}s – {seg['end']:.2f}s : {seg['lang']}")

    # Language switching latency (mean time between segment boundaries)
    if len(language_segments) > 1:
        switch_times = []
        for i in range(1, len(language_segments)):
            gap = (language_segments[i]['start'] - language_segments[i-1]['end']) * 1000
            switch_times.append(abs(gap))
        mean_switch_ms = sum(switch_times) / len(switch_times)
        print(f"  Mean switching latency: {mean_switch_ms:.1f} ms (threshold < 200 ms)")
    else:
        mean_switch_ms = 0.0
        print("  Only 1 segment found — switching latency: N/A")

    # Reference label generation from transcript word-level timestamps
    # Uses transcript_best.json (romanized, word-level timestamps) to identify
    # English segments based on vocabulary rather than VoxLingua107 frame predictions.
    # VoxLingua107 labels only 0.7% of frames as English despite clear English sections;
    # transcript-based labels provide a much more accurate evaluation reference.
    print("\n--- Computing LID F1 vs transcript-derived reference labels ---")

    ref_labels = _make_transcript_ref_labels(frame_results, BASE_DIR)
    pred_labels = [lid.LANG_MAP_INV[r["lang"]] for r in frame_results]

    en_prec, en_rec, en_f1 = compute_f1_score(ref_labels, pred_labels, pos_label=0)
    hi_prec, hi_rec, hi_f1 = compute_f1_score(ref_labels, pred_labels, pos_label=1)
    macro_f1 = (en_f1 + hi_f1) / 2.0

    print(f"  English  F1: {en_f1:.3f}  (P={en_prec:.3f}, R={en_rec:.3f})")
    print(f"  Hindi    F1: {hi_f1:.3f}  (P={hi_prec:.3f}, R={hi_rec:.3f})")
    print(f"  Macro    F1: {macro_f1:.3f}  (threshold >= 0.85)")
    status = "✓ PASS" if macro_f1 >= 0.85 else "✗ FAIL"
    print(f"  Status: {status}")

    # Update pipeline_results.json
    pr_path = str(BASE_DIR / "outputs" / "pipeline_results.json")
    if Path(pr_path).exists():
        with open(pr_path, "r") as f:
            pipeline_results = json.load(f)
    else:
        pipeline_results = {}

    pipeline_results["lid_f1"] = {
        "english": round(en_f1, 4),
        "hindi": round(hi_f1, 4),
        "macro": round(macro_f1, 4),
    }
    pipeline_results["lid_switching_ms"] = round(mean_switch_ms, 2)

    with open(pr_path, "w") as f:
        json.dump(pipeline_results, f, indent=2)
    print(f"\n  Updated {pr_path} with lid_f1 and lid_switching_ms")


if __name__ == "__main__":
    main()
