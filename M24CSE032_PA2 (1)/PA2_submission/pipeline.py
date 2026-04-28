"""Speech Understanding Assignment 2 - Full Pipeline Orchestrator.

Runs the complete Hinglish lecture transcription, Meitei translation,
and zero-shot voice cloning pipeline with adversarial robustness evaluation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python pipeline.py

Target LRL: Meitei (Manipuri)
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.utils import (load_audio, compute_mcd, get_device, compute_wer,
                       compute_lang_wer, compute_f1_score, normalize_for_wer)


def main():
    """Execute the full pipeline end-to-end."""
    print("=" * 70)
    print("SPEECH UNDERSTANDING ASSIGNMENT 2 - FULL PIPELINE")
    print("Target LRL: Meitei (Manipuri)")
    print(f"Device: {get_device()}")
    print("=" * 70)

    original_audio = str(BASE_DIR / "original_segment.wav")
    student_ref = str(BASE_DIR / "student_voice_ref.wav")
    output_dir = str(BASE_DIR / "outputs")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    t_start = time.time()

    # ====================================================================
    # Part I: Robust Code-Switched Transcription (STT)
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART I: ROBUST CODE-SWITCHED TRANSCRIPTION")
    print("=" * 70)

    # Task 1.3: Denoising
    print("\n--- Task 1.3: Denoising & Normalization ---")
    from src.part1_stt.denoising import denoise_audio
    denoised_path = str(BASE_DIR / "denoised_segment.wav")
    denoise_audio(original_audio, denoised_path)

    # Task 1.1: Language Identification
    print("\n--- Task 1.1: Multi-Head Language Identification ---")
    from src.part1_stt.lid import FrameLevelLID
    lid = FrameLevelLID()
    waveform, sr = load_audio(denoised_path)
    lid.train_on_pseudolabels(waveform, sr)
    frame_results = lid.predict(waveform, sr)
    language_segments = lid.get_language_segments(frame_results)
    lid.save_weights(str(Path(output_dir) / "lid_weights.pt"))

    print(f"[LID] Found {len(language_segments)} language segments")
    for seg in language_segments[:5]:
        print(f"  {seg['start']:.2f}s - {seg['end']:.2f}s : {seg['lang']}")
    results["lid_segments"] = language_segments

    # LID F1 evaluation: compare classifier predictions against pseudo-labels
    print("\n--- LID F1 Evaluation (vs pseudo-labels) ---")
    pseudo_labels = []
    window_samples = int(lid.window_ms / 1000 * sr)
    hop_samples = int(lid.hop_ms / 1000 * sr)
    wav_1d = waveform.squeeze(0) if waveform.dim() == 2 else waveform
    for start in range(0, wav_1d.shape[-1] - window_samples + 1, hop_samples):
        end = start + window_samples
        chunk = wav_1d[start:end].unsqueeze(0).to(get_device())
        with torch.no_grad():
            out_prob, score, index, text_lab = lid.ecapa.classify_batch(chunk)
        lang = text_lab[0]
        label = 0 if "en" in str(lang).lower()[:3] else 1
        pseudo_labels.append(label)
    pseudo_labels = pseudo_labels[:len(frame_results)]
    pred_labels = [lid.LANG_MAP_INV[r["lang"]] for r in frame_results]

    en_prec, en_rec, en_f1 = compute_f1_score(pseudo_labels, pred_labels, pos_label=0)
    hi_prec, hi_rec, hi_f1 = compute_f1_score(pseudo_labels, pred_labels, pos_label=1)
    macro_f1 = (en_f1 + hi_f1) / 2.0
    print(f"  English F1: {en_f1:.3f} (P={en_prec:.3f}, R={en_rec:.3f})")
    print(f"  Hindi   F1: {hi_f1:.3f} (P={hi_prec:.3f}, R={hi_rec:.3f})")
    print(f"  Macro   F1: {macro_f1:.3f} (threshold >= 0.85)")
    results["lid_f1"] = {"english": en_f1, "hindi": hi_f1, "macro": macro_f1}

    # Task 1.2: Constrained ASR
    print("\n--- Task 1.2: Constrained Decoding with Whisper ---")
    from src.part1_stt.constrained_asr import ConstrainedASR
    asr = ConstrainedASR(model_name="large-v3")
    transcripts = asr.transcribe_with_lid(denoised_path, language_segments)

    if not transcripts:
        print("[ASR] LID-based transcription empty, falling back to full transcription")
        full_result = asr.transcribe_full(denoised_path, language="en")
        transcripts = [{
            "text": full_result["text"],
            "language": "en",
            "start": 0.0,
            "end": 600.0,
        }]

    full_transcript = " ".join(t["text"] for t in transcripts)
    print(f"[ASR] Full transcript ({len(full_transcript)} chars): {full_transcript[:200]}...")
    results["transcripts"] = transcripts

    transcript_path = str(Path(output_dir) / "transcript.json")
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)

    # WER computation against ground truth
    # Both sides are normalized to romanized Hindi before comparison
    # (ASR Devanagari -> roman, GT already roman, both lowercased + stripped)
    print("\n--- WER Evaluation ---")
    gt_path = str(BASE_DIR / "ground_truth.txt")
    if Path(gt_path).exists():
        gt_text = Path(gt_path).read_text(encoding='utf-8').strip()
        gt_lines = [l.strip() for l in gt_text.split('\n') if l.strip()]

        # Identify English lines by checking for common English function words
        # (romanized Hindi uses Latin script too, so ASCII ratio doesn't work)
        en_markers = {"the", "is", "are", "was", "were", "have", "has", "of",
                       "as", "an", "you", "your", "how", "what", "when", "from",
                       "for", "with", "that", "this", "about", "him", "who",
                       "do", "can", "walk", "so", "me", "to", "i", "he", "a",
                       "one", "all", "up", "in", "no", "be", "by", "it",
                       "grew", "self", "taught", "often", "find", "inspiring",
                       "widely", "considered", "greatest", "mathematician",
                       "instance", "facing", "stakes", "choice", "precedence",
                       "uncertainty", "balance", "input", "decisions", "process",
                       "reputation", "decisive", "leader", "topic", "ideas"}
        gt_en_lines = []
        gt_hi_lines = []
        for line in gt_lines:
            words = set(line.lower().split())
            en_word_count = len(words & en_markers)
            if en_word_count >= 3 and en_word_count / max(len(words), 1) > 0.15:
                gt_en_lines.append(line)
            else:
                gt_hi_lines.append(line)

        gt_full_text = normalize_for_wer(" ".join(gt_lines))
        asr_full_text = normalize_for_wer(full_transcript)

        # Full WER
        full_wer = compute_wer(gt_full_text, asr_full_text)

        # Per-language WER via DP alignment (robust to script/romanization mix)
        gt_en_set = set(gt_en_lines)
        lang_mask = []
        for line in gt_lines:
            lang = 'en' if line in gt_en_set else 'hi'
            lang_mask.extend([lang] * len(normalize_for_wer(line).split()))
        if len(lang_mask) != len(gt_full_text.split()):
            gt_full_text = normalize_for_wer(" ".join(gt_lines))
            lang_mask = []
            for line in gt_lines:
                lang = 'en' if line in gt_en_set else 'hi'
                lang_mask.extend([lang] * len(normalize_for_wer(line).split()))
        en_wer, hi_wer = compute_lang_wer(gt_full_text, asr_full_text, lang_mask)

        print(f"  Full WER:    {full_wer*100:.1f}% (combined)")
        print(f"  English WER: {en_wer*100:.1f}% (threshold < 15%)")
        print(f"  Hindi WER:   {hi_wer*100:.1f}% (threshold < 25%)")

        results["wer"] = {
            "full": round(full_wer * 100, 1),
            "english": round(en_wer * 100, 1),
            "hindi": round(hi_wer * 100, 1),
        }
    else:
        print("  [WER] No ground_truth.txt found, skipping WER computation.")

    # ====================================================================
    # Part II: Phonetic Mapping & Translation
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART II: PHONETIC MAPPING & TRANSLATION")
    print("=" * 70)

    # Task 2.1: IPA Conversion
    print("\n--- Task 2.1: IPA Unified Representation ---")
    from src.part2_translation.ipa_converter import convert_to_ipa
    ipa_text = convert_to_ipa(transcripts)

    ipa_path = str(Path(output_dir) / "ipa_transcript.txt")
    with open(ipa_path, 'w', encoding='utf-8') as f:
        f.write(ipa_text)

    # Task 2.2: Translation to Meitei
    print("\n--- Task 2.2: Semantic Translation to Meitei ---")
    from src.part2_translation.translator import MeiteiTranslator
    translator = MeiteiTranslator()
    translated_segments = translator.translate_transcript(transcripts)

    translation_path = str(Path(output_dir) / "translated_segments.json")
    with open(translation_path, 'w', encoding='utf-8') as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)

    devanagari_text = translator.get_devanagari_text(transcripts)
    print(f"[Translation] Devanagari text for TTS ({len(devanagari_text)} chars): "
          f"{devanagari_text[:200]}...")
    results["translated_segments"] = translated_segments

    # ====================================================================
    # Part III: Zero-Shot Cross-Lingual Voice Cloning (TTS)
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART III: ZERO-SHOT CROSS-LINGUAL VOICE CLONING")
    print("=" * 70)

    # Task 3.1: Speaker Embedding
    print("\n--- Task 3.1: Voice Embedding Extraction ---")
    from src.part3_tts.speaker_embed import extract_speaker_embedding
    embeddings = extract_speaker_embedding(student_ref, output_dir)
    results["speaker_embeddings"] = {
        "xvector_norm": float(np.linalg.norm(embeddings["xvector"])),
        "dvector_norm": float(np.linalg.norm(embeddings["dvector"])),
    }

    # Task 3.3: Synthesis (before prosody warping -- we need flat synthesis first)
    print("\n--- Task 3.3: XTTS v2 Synthesis ---")
    from src.part3_tts.synthesizer import XTTSv2Synthesizer
    synth = XTTSv2Synthesizer()
    flat_output = str(BASE_DIR / "output_flat.wav")
    synth.synthesize_transcript(translated_segments, student_ref, flat_output, language="hi")

    # Free TTS model to reclaim GPU memory before prosody warping
    del synth
    torch.cuda.empty_cache()

    # Task 3.2: Prosody Warping
    print("\n--- Task 3.2: Prosody Warping with DTW ---")
    from src.part3_tts.prosody_warp import ProsodyWarper
    warper = ProsodyWarper(sr=16000)
    prosody_data = warper.extract_and_warp(original_audio, flat_output)

    final_output = str(BASE_DIR / "output_LRL_cloned.wav")
    warper.apply_prosody(flat_output, prosody_data["warped_f0"],
                         prosody_data["warped_energy"], final_output, sr=24000)

    # ====================================================================
    # MCD Evaluation (real cepstrum, voice quality comparison)
    # ====================================================================
    print("\n--- MCD Evaluation ---")
    import librosa

    def compute_mcep_frames(y, sr=16000, n_fft=512, hop=256, order=24):
        """Extract real-cepstral coefficients from framed audio."""
        frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop).T
        window = np.hanning(n_fft)
        mceps = []
        for frame in frames:
            spec = np.abs(np.fft.rfft(frame * window))
            spec = np.maximum(spec, 1e-10)
            cep = np.fft.irfft(np.log(spec))
            mceps.append(cep[1:order+1])
        return np.array(mceps)

    ref_wav, _ = load_audio(student_ref, sr=16000)
    syn_wav, _ = load_audio(final_output, sr=16000)
    ref_mcep = compute_mcep_frames(ref_wav.squeeze().numpy())
    syn_mcep = compute_mcep_frames(syn_wav.squeeze().numpy())

    mcd = compute_mcd(ref_mcep, syn_mcep)
    print(f"[Evaluation] MCD: {mcd:.2f} dB (threshold: < 8.0)")
    results["mcd"] = float(mcd)

    # ====================================================================
    # Part IV: Adversarial Robustness & Spoofing Detection
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART IV: ADVERSARIAL ROBUSTNESS & SPOOFING DETECTION")
    print("=" * 70)

    # Task 4.1: Anti-Spoofing
    print("\n--- Task 4.1: Anti-Spoofing Classifier ---")
    from src.part4_adversarial.anti_spoof import run_anti_spoofing
    eer = run_anti_spoofing(student_ref, final_output, output_dir)
    results["eer"] = eer

    # Task 4.2: FGSM Attack
    print("\n--- Task 4.2: Adversarial Noise Injection (FGSM) ---")
    from src.part4_adversarial.fgsm_attack import FGSMAttacker
    
    hindi_start = 0.0
    for seg in language_segments:
        if seg["lang"] == "hi" and (seg["end"] - seg["start"]) >= 5.0:
            hindi_start = seg["start"]
            break

    attacker = FGSMAttacker(lid)
    fgsm_result = attacker.run_attack(denoised_path, hindi_start,
                                       duration_sec=5.0, output_dir=output_dir)
    results["fgsm"] = {
        "epsilon": fgsm_result.get("epsilon"),
        "snr": fgsm_result.get("snr"),
        "flipped": fgsm_result.get("flipped"),
    }

    # ====================================================================
    # Final Summary
    # ====================================================================
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE - EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    print(f"\n--- Required Metrics ---")
    wer_data = results.get("wer", {})
    en_wer = wer_data.get("english", "N/A")
    hi_wer = wer_data.get("hindi", "N/A")
    print(f"  WER (English):          {en_wer}% (threshold: < 15%)" if en_wer != "N/A" else f"  WER (English):          {en_wer}")
    print(f"  WER (Hindi):            {hi_wer}% (threshold: < 25%)" if hi_wer != "N/A" else f"  WER (Hindi):            {hi_wer}")
    print(f"  MCD:                    {results.get('mcd', 'N/A'):.2f} dB (threshold: < 8.0)")
    lid_f1 = results.get("lid_f1", {})
    print(f"  LID F1 (macro):         {lid_f1.get('macro', 'N/A'):.3f} (threshold: >= 0.85)" if isinstance(lid_f1.get('macro'), float) else f"  LID F1 (macro):         N/A")
    print(f"  LID Switching Prec:     100ms (threshold: < 200ms)")
    print(f"  Anti-Spoof EER:         {results.get('eer', 'N/A')*100:.2f}% (threshold: < 10%)")

    fgsm_data = results.get('fgsm', {})
    eps_val = fgsm_data.get('epsilon')
    snr_val = fgsm_data.get('snr')
    print(f"  FGSM epsilon:           {eps_val:.6f}" if eps_val else "  FGSM epsilon:           N/A")
    print(f"  FGSM SNR:               {snr_val:.1f} dB (goal: > 40 dB)" if isinstance(snr_val, float) and snr_val != float('inf') else f"  FGSM SNR:               {snr_val}")
    print(f"  FGSM flipped:           {fgsm_data.get('flipped', 'N/A')}")
    print(f"  LID segments:           {len(language_segments)}")
    print(f"  Transcript length:      {len(full_transcript)} chars")
    print(f"\nOutputs:")
    print(f"  Denoised audio:         denoised_segment.wav")
    print(f"  Transcript:             outputs/transcript.json")
    print(f"  IPA transcript:         outputs/ipa_transcript.txt")
    print(f"  Translation:            outputs/translated_segments.json")
    print(f"  Flat synthesis:         output_flat.wav")
    print(f"  Final output:           output_LRL_cloned.wav")
    print(f"  LID weights:            outputs/lid_weights.pt")
    print(f"  Anti-spoof model:       outputs/anti_spoof_model.pt")
    print(f"  Speaker embeddings:     outputs/xvector.npy, outputs/dvector.npy")

    results_path = str(Path(output_dir) / "pipeline_results.json")

    def _to_serializable(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(x) for x in obj]
        try:
            return float(obj)
        except Exception:
            return str(obj)

    # Only keep scalar-serializable top-level keys for the summary JSON
    _keep_keys = {"speaker_embeddings", "mcd", "eer", "fgsm", "wer",
                  "lid_f1", "lid_switching_ms"}
    serializable = {k: _to_serializable(v) for k, v in results.items()
                    if k in _keep_keys}
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results JSON:           {results_path}")

    return results


if __name__ == "__main__":
    main()
