# Speech Understanding - Programming Assignment 2

**Roll No:** M24CSE032  
**Course:** Speech Understanding — Programming Assignment 2

## Hinglish Lecture Transcription & Meitei Voice Cloning Pipeline

This project implements a complete pipeline that:

1. **Transcribes** a 10-minute Hinglish (code-switched English-Hindi) lecture
2. **Translates** the transcript to Meitei (Manipuri) — a low-resource language
3. **Synthesizes** the lecture in Meitei using zero-shot voice cloning with the student's voice
4. **Evaluates** adversarial robustness and spoofing detection

### Target LRL: Meitei (Manipuri) — ISO 639-3: `mni`

---

## Setup

```bash
# Create and activate conda environment
conda create -n speech_hw2 python=3.10 -y
conda activate speech_hw2

# Install PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install -r requirements.txt
```

## Place Your Audio Files

Before running, place the following files in the root directory:
- `original_segment.wav` — your 10-minute classroom lecture (16 kHz, mono)
- `student_voice_ref.wav` — your 60-second voice reference recording

## Run the Full Pipeline

```bash
CUDA_VISIBLE_DEVICES=0 python pipeline.py
```

## Project Structure

```
PA2_submission/
  pipeline.py                        # Main orchestrator — runs all 4 parts
  retrain_lid.py                     # Re-train LID model standalone
  eval_eer.py                        # Evaluate anti-spoofing EER
  eval_fgsm.py                       # Evaluate FGSM adversarial attack
  eval_lid.py                        # Evaluate LID F1
  eval_wer.py                        # Evaluate WER
  requirements.txt                   # Python dependencies
  ground_truth.txt                   # Reference transcript for WER
  report.md                          # 10-page IEEE-style report
  implementation_note.md             # 1-page non-obvious design choices

  src/
    utils.py                         # Shared utilities (load_audio, MCD, WER)

    part1_stt/
      denoising.py                   # Task 1.3: Spectral Subtraction denoiser
      lid.py                         # Task 1.1: Multi-Head Attention LID (ECAPA-TDNN)
      constrained_asr.py             # Task 1.2: Whisper + N-gram Logit Bias
      ngram_lm.py                    # N-gram Language Model from syllabus
      syllabus_corpus.txt            # Speech course technical terms

    part2_translation/
      ipa_converter.py               # Task 2.1: Hinglish IPA converter
      translator.py                  # Task 2.2: Meitei translation
      parallel_corpus.json           # 500-word EN/HI → Meitei dictionary

    part3_tts/
      speaker_embed.py               # Task 3.1: x-vector & d-vector extraction
      prosody_warp.py                # Task 3.2: F0/Energy DTW warping
      synthesizer.py                 # Task 3.3: XTTS v2 zero-shot synthesis

    part4_adversarial/
      anti_spoof.py                  # Task 4.1: LFCC+CQCC anti-spoofing CM
      fgsm_attack.py                 # Task 4.2: FGSM adversarial attack

  configs/
    environment.yaml                 # Environment configuration

  outputs/                           # Generated during pipeline run
    transcript.json
    ipa_transcript.txt
    translated_segments.json
    lid_weights.pt
    anti_spoof_model.pt
    pipeline_results.json
```

## Evaluation Metrics

| Metric | Threshold | Status |
|---|---|---|
| WER (English) | < 15% | Evaluated in pipeline |
| WER (Hindi) | < 25% | Evaluated in pipeline |
| MCD | < 8.0 dB | Evaluated in pipeline |
| LID Switching | < 200ms | Evaluated in pipeline |
| EER (Anti-Spoof) | < 10% | Evaluated in pipeline |
| FGSM epsilon | Report value | Evaluated in pipeline |

## Key Libraries

- `openai-whisper` — Whisper-large-v3 ASR
- `speechbrain` — ECAPA-TDNN for LID and speaker embeddings
- `TTS (Coqui)` — XTTS v2 for zero-shot voice cloning
- `epitran` — Grapheme-to-phoneme conversion
- `librosa`, `parselmouth` — Audio analysis and PSOLA prosody
- `resemblyzer` — d-vector extraction
- `torch`, `torchaudio` — Core deep learning
