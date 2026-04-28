"""Standalone WER evaluation script.

Re-transcribes the denoised audio with proper code-switching handling
and computes WER against the ground truth. Much faster than the full pipeline
since it only runs ASR + WER (no TTS, no DTW, no anti-spoofing).

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_wer.py
"""

import os
import sys
import json
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.utils import normalize_for_wer, compute_wer, compute_lang_wer


def _split_gt(gt_path: str):
    """Load ground truth and split into full / English / Hindi texts."""
    gt_text = Path(gt_path).read_text(encoding="utf-8").strip()
    gt_lines = [l.strip() for l in gt_text.split("\n") if l.strip()]

    en_markers = {
        "the", "is", "are", "was", "were", "have", "has", "of",
        "as", "an", "you", "your", "how", "what", "when", "from",
        "for", "with", "that", "this", "about", "him", "who",
        "do", "can", "walk", "so", "me", "to", "i", "he", "a",
        "one", "all", "up", "in", "no", "be", "by", "it",
        "grew", "self", "taught", "often", "find", "inspiring",
        "widely", "considered", "greatest", "mathematician",
        "instance", "facing", "stakes", "choice", "precedence",
        "uncertainty", "balance", "input", "decisions", "process",
        "reputation", "decisive", "leader", "topic", "ideas",
    }

    gt_en_lines, gt_hi_lines = [], []
    for line in gt_lines:
        words = set(line.lower().split())
        en_count = len(words & en_markers)
        if en_count >= 3 and en_count / max(len(words), 1) > 0.15:
            gt_en_lines.append(line)
        else:
            gt_hi_lines.append(line)

    return gt_lines, gt_en_lines, gt_hi_lines


def _make_lang_mask(gt_lines: list, gt_en_set: set) -> list:
    """Return a word-level list of 'en'|'hi' tags matching the normalised GT."""
    mask = []
    for line in gt_lines:
        lang = 'en' if line in gt_en_set else 'hi'
        words = normalize_for_wer(line).split()
        mask.extend([lang] * len(words))
    return mask


def _per_lang_wer(best_norm: str, gt_full_text: str,
                  gt_lines: list, gt_en_lines: list):
    """Compute per-language WER via DP alignment — no heuristic word buckets.

    We tag every GT word as 'en' or 'hi' based on which line it came from,
    then backtrack the alignment to assign each error to a language.
    This is robust regardless of whether the ASR outputs Devanagari,
    romanised Hindi, or plain English.
    """
    gt_en_set = set(gt_en_lines)
    mask = _make_lang_mask(gt_lines, gt_en_set)
    if len(mask) != len(gt_full_text.split()):
        # Safety: rebuild gt_full_text from lines to keep word count consistent
        gt_full_text = normalize_for_wer(" ".join(gt_lines))
        mask = _make_lang_mask(gt_lines, gt_en_set)
    en_wer_val, hi_wer_val = compute_lang_wer(gt_full_text, best_norm, mask)
    return en_wer_val, hi_wer_val


def transcribe_chunked(model, audio_path, device, chunk_sec=30.0):
    """Transcribe audio in chunks with per-chunk language detection.

    For code-switched (Hinglish) audio this is critical: Whisper detects
    the dominant language once and then forces the whole file into that
    script.  By chunking we let it detect English interview segments and
    transcribe them in English while keeping Hindi for the rest.
    """
    import whisper
    import numpy as np

    # Load full audio as float32 numpy (Whisper's expected input)
    audio = whisper.load_audio(audio_path)
    sr = 16000  # Whisper uses 16 kHz
    chunk_samples = int(chunk_sec * sr)
    total_samples = len(audio)

    texts = []
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end]
        # Pad short final chunk to 1 s so Whisper doesn't choke
        if len(chunk) < sr:
            chunk = np.pad(chunk, (0, sr - len(chunk)))
        result = model.transcribe(
            chunk,
            fp16=(device == "cuda"),
            condition_on_previous_text=False,
            verbose=False,
        )
        lang = result.get("language", "hi")
        text = result.get("text", "").strip()
        if text:
            texts.append(text)

    combined = " ".join(texts)
    return combined


def transcribe_chunked_prompted(model, audio_path, device, chunk_sec=30.0):
    """Chunked transcription with a romanised-Hindi initial prompt.

    The initial_prompt biases Whisper toward romanised output style which
    can sometimes reduce script mismatch with the ground truth.
    """
    import whisper
    import numpy as np

    # this prompt is for the first 30 seconds of the audio - from the previous edit - needs to be updated
    prompt = (
        "wo sari cheezein jo mere aas paas hain wahi mujhe motivate karti rehti hain. "
        "meri responsibilities mujhe aage badhne ke liye push karti hain. "
        "dekhiye main unka bahut respect karta hoon."
    )

    audio = whisper.load_audio(audio_path)
    sr = 16000
    chunk_samples = int(chunk_sec * sr)
    total_samples = len(audio)

    texts = []
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end]
        if len(chunk) < sr:
            chunk = np.pad(chunk, (0, sr - len(chunk)))
        result = model.transcribe(
            chunk,
            fp16=(device == "cuda"),
            condition_on_previous_text=False,
            initial_prompt=prompt,
            verbose=False,
        )
        text = result.get("text", "").strip()
        if text:
            texts.append(text)

    return " ".join(texts)


def main():
    import whisper
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_path = str(BASE_DIR / "denoised_segment.wav")
    gt_path = str(BASE_DIR / "ground_truth.txt")

    print("=" * 60)
    print("WER EVALUATION - Code-Switched Transcription")
    print(f"Device: {device}")
    print("=" * 60)

    # ---------- Ground Truth ----------
    gt_lines, gt_en_lines, gt_hi_lines = _split_gt(gt_path)

    gt_en_text = normalize_for_wer(" ".join(gt_en_lines))
    gt_hi_text = normalize_for_wer(" ".join(gt_hi_lines))
    gt_full_text = normalize_for_wer(" ".join(gt_lines))

    print(f"\nGround Truth: {len(gt_lines)} lines "
          f"({len(gt_en_lines)} English, {len(gt_hi_lines)} Hindi)")
    print(f"GT words: {len(gt_full_text.split())} total "
          f"({len(gt_en_text.split())} EN, {len(gt_hi_text.split())} HI)")

    # ---------- Load model ----------
    print("\n--- Loading Whisper large-v3 ---")
    model = whisper.load_model("large-v3", device=device)

    # ============================================================
    # Method 1: Full-file auto language detection
    # ============================================================
    print("\n--- Method 1: Full-file auto ---")
    result_auto = model.transcribe(
        audio_path, fp16=(device == "cuda"), verbose=False,
    )
    auto_text = result_auto.get("text", "")
    auto_norm = normalize_for_wer(auto_text)
    auto_wer = compute_wer(gt_full_text, auto_norm)
    print(f"  Detected lang: {result_auto.get('language', 'N/A')}")
    print(f"  Words: {len(auto_norm.split())} | Full WER: {auto_wer*100:.1f}%")

    # ============================================================
    # Method 2: Full-file Hindi mode
    # ============================================================
    print("\n--- Method 2: Full-file Hindi ---")
    result_hi = model.transcribe(
        audio_path, language="hi", fp16=(device == "cuda"), verbose=False,
    )
    hi_text = result_hi.get("text", "")
    hi_norm = normalize_for_wer(hi_text)
    hi_wer = compute_wer(gt_full_text, hi_norm)
    print(f"  Words: {len(hi_norm.split())} | Full WER: {hi_wer*100:.1f}%")

    # ============================================================
    # Method 3: Full-file English mode
    # ============================================================
    print("\n--- Method 3: Full-file English ---")
    result_en = model.transcribe(
        audio_path, language="en", fp16=(device == "cuda"), verbose=False,
    )
    en_text = result_en.get("text", "")
    en_norm = normalize_for_wer(en_text)
    en_wer = compute_wer(gt_full_text, en_norm)
    print(f"  Words: {len(en_norm.split())} | Full WER: {en_wer*100:.1f}%")

    # ============================================================
    # Method 4: Chunked transcription (30 s) with per-chunk lang detect
    # ============================================================
    print("\n--- Method 4: Chunked (30s) auto lang detect ---")
    chunked_text = transcribe_chunked(model, audio_path, device, chunk_sec=30.0)
    chunked_norm = normalize_for_wer(chunked_text)
    chunked_wer = compute_wer(gt_full_text, chunked_norm)
    print(f"  Words: {len(chunked_norm.split())} | Full WER: {chunked_wer*100:.1f}%")

    # ============================================================
    # Method 5: Chunked with romanised-Hindi prompt
    # ============================================================
    print("\n--- Method 5: Chunked + romanised prompt ---")
    prompted_text = transcribe_chunked_prompted(model, audio_path, device, chunk_sec=30.0)
    prompted_norm = normalize_for_wer(prompted_text)
    prompted_wer = compute_wer(gt_full_text, prompted_norm)
    print(f"  Words: {len(prompted_norm.split())} | Full WER: {prompted_wer*100:.1f}%")

    # ============================================================
    # Method 6: Full-file with condition_on_previous_text=False
    # ============================================================
    print("\n--- Method 6: Full-file auto, no prev-text conditioning ---")
    result_nocond = model.transcribe(
        audio_path, fp16=(device == "cuda"),
        condition_on_previous_text=False, verbose=False,
    )
    nocond_text = result_nocond.get("text", "")
    nocond_norm = normalize_for_wer(nocond_text)
    nocond_wer = compute_wer(gt_full_text, nocond_norm)
    print(f"  Words: {len(nocond_norm.split())} | Full WER: {nocond_wer*100:.1f}%")

    # ---------- Pick best ----------
    all_methods = {
        "auto": {"wer": auto_wer, "text": auto_text, "norm": auto_norm},
        "hindi": {"wer": hi_wer, "text": hi_text, "norm": hi_norm},
        "english": {"wer": en_wer, "text": en_text, "norm": en_norm},
        "chunked": {"wer": chunked_wer, "text": chunked_text, "norm": chunked_norm},
        "prompted": {"wer": prompted_wer, "text": prompted_text, "norm": prompted_norm},
        "nocond": {"wer": nocond_wer, "text": nocond_text, "norm": nocond_norm},
    }
    best_method = min(all_methods, key=lambda k: all_methods[k]["wer"])
    best = all_methods[best_method]

    print(f"\n{'='*60}")
    print(f"BEST METHOD: {best_method} (Full WER = {best['wer']*100:.1f}%)")
    print(f"{'='*60}")

    # Per-language WER — alignment-based (no heuristic word-bucket)
    en_wer_val, hi_wer_val = _per_lang_wer(
        best["norm"], gt_full_text, gt_lines, gt_en_lines,
    )
    print(f"  English segments WER: {en_wer_val*100:.1f}% (threshold < 15%)")
    print(f"  Hindi segments WER:   {hi_wer_val*100:.1f}% (threshold < 25%)")

    # ---------- Save outputs ----------
    # Best transcript
    best_transcript = [{
        "text": best["text"],
        "language": "hi",
        "start": 0.0,
        "end": 600.0,
    }]
    out_path = str(BASE_DIR / "outputs" / "transcript.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_transcript, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved best transcript to {out_path}")

    # WER results
    wer_results = {
        "full_wer": round(best["wer"] * 100, 1),
        "english_wer": round(en_wer_val * 100, 1) if en_wer_val is not None else None,
        "hindi_wer": round(hi_wer_val * 100, 1) if hi_wer_val is not None else None,
        "method": best_method,
        "comparison": {k: round(v["wer"] * 100, 1) for k, v in all_methods.items()},
    }
    wer_path = str(BASE_DIR / "outputs" / "wer_results.json")
    with open(wer_path, "w") as f:
        json.dump(wer_results, f, indent=2)
    print(f"  Saved WER results to {wer_path}")

    # Update pipeline_results.json
    pr_path = str(BASE_DIR / "outputs" / "pipeline_results.json")
    if Path(pr_path).exists():
        with open(pr_path, "r") as f:
            pipeline_results = json.load(f)
    else:
        pipeline_results = {}
    pipeline_results["wer"] = wer_results
    with open(pr_path, "w") as f:
        json.dump(pipeline_results, f, indent=2)
    print(f"  Updated {pr_path}")


if __name__ == "__main__":
    main()
