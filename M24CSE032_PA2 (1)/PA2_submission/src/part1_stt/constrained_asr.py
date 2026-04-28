"""Task 1.2: Constrained Decoding with Whisper + N-gram Logit Bias.

Uses Whisper-large-v3 with custom logit biasing from an N-gram language model
trained on the speech course syllabus to prioritize technical terms.

The logit bias is injected into Whisper's decode loop via a custom LogitFilter
that is appended to DecodingTask.logit_filters through monkey-patching.
"""

import torch
import whisper
import whisper.decoding as whisper_decoding
import numpy as np
import re
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import load_audio, get_device
from src.part1_stt.ngram_lm import NgramLanguageModel


class NgramLogitFilter:
    """Whisper-compatible LogitFilter that applies N-gram LM bias at each decode step.

    Implements the same interface as whisper.decoding.SuppressTokens etc.:
        apply(logits: Tensor, tokens: Tensor) -> None   (in-place modification)

    At each step it adds:
        1. A static boost for every vocabulary token that appears in the syllabus corpus.
        2. A contextual trigram boost conditioned on the last 2-3 decoded tokens.
    """

    def __init__(self, ngram_lm: NgramLanguageModel, tokenizer,
                 alpha: float = 0.3, max_boost: float = 5.0):
        self.ngram_lm = ngram_lm
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.max_boost = max_boost
        self._build_token_boost_map()

    def _build_token_boost_map(self):
        """Pre-compute boost scores for all tokens in Whisper's vocabulary."""
        self.token_boosts = {}
        vocab_size = 51865
        for token_id in range(vocab_size):
            try:
                token_text = self.tokenizer.decode([token_id]).strip().lower()
            except Exception:
                continue

            words = re.findall(r'[a-z]{2,}', token_text)
            if words:
                boost = sum(self.ngram_lm.get_term_boost(w) for w in words) / len(words)
                if boost > 0:
                    self.token_boosts[token_id] = min(boost * self.alpha, self.max_boost)

        print(f"[LogitBias] Built boost map for {len(self.token_boosts)} tokens")

    def apply(self, logits: torch.Tensor, tokens: torch.Tensor) -> None:
        """Apply N-gram logit bias in-place (Whisper LogitFilter interface).

        Args:
            logits: (batch, vocab_size) tensor of raw logits at current step
            tokens: (batch, seq_len) tensor of previously decoded token IDs
        """
        # --- static term boost ---
        for token_id, boost in self.token_boosts.items():
            if token_id < logits.shape[-1]:
                logits[:, token_id] += boost

        # --- contextual trigram boost ---
        if tokens.shape[-1] >= 2:
            # Use the last 3 tokens as context (first batch element is representative)
            context_ids = tokens[0, -3:].tolist()
            context_words = []
            for tid in context_ids:
                try:
                    w = self.tokenizer.decode([tid]).strip().lower()
                    context_words.extend(re.findall(r'[a-z]+', w))
                except Exception:
                    pass

            if context_words:
                for token_id, base_boost in self.token_boosts.items():
                    try:
                        token_text = self.tokenizer.decode([token_id]).strip().lower()
                        token_words = re.findall(r'[a-z]+', token_text)
                        if token_words:
                            ngram_score = self.ngram_lm.log_prob(token_words[0], context_words)
                            contextual_boost = self.alpha * max(ngram_score + 5, 0)
                            if contextual_boost > 0 and token_id < logits.shape[-1]:
                                logits[:, token_id] += min(contextual_boost, self.max_boost)
                    except Exception:
                        pass


@contextmanager
def inject_logit_filter(logit_filter):
    """Context manager that monkey-patches DecodingTask.__init__ to append
    our NgramLogitFilter into Whisper's decode loop.

    Every DecodingTask created inside this context will have the filter
    appended to its logit_filters list, so it is applied at every decode
    step during model.transcribe().
    """
    original_init = whisper_decoding.DecodingTask.__init__

    def patched_init(self_task, model, options):
        original_init(self_task, model, options)
        self_task.logit_filters.append(logit_filter)

    whisper_decoding.DecodingTask.__init__ = patched_init
    try:
        yield
    finally:
        whisper_decoding.DecodingTask.__init__ = original_init


class ConstrainedASR:
    """Whisper-based ASR with constrained decoding using N-gram logit bias."""

    def __init__(self, model_name: str = "large-v3", device: Optional[torch.device] = None):
        self.device = device or get_device()
        print(f"[ASR] Loading Whisper {model_name}...")
        self.model = whisper.load_model(model_name, device=str(self.device))
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

        syllabus_path = Path(__file__).parent / "syllabus_corpus.txt"
        self.ngram_lm = NgramLanguageModel(n=3)
        self.ngram_lm.train(str(syllabus_path))

        self.logit_filter = NgramLogitFilter(
            self.ngram_lm, self.tokenizer, alpha=0.3
        )

    def transcribe_segment(self, waveform: torch.Tensor, sr: int = 16000,
                           language: str = "en") -> Dict:
        """Transcribe a single segment with constrained decoding.

        Injects the NgramLogitFilter into Whisper's DecodingTask so that
        N-gram logit bias is applied at every decode step.
        """
        import tempfile, soundfile as sf

        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        audio_np = waveform.numpy()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio_np, sr)

        try:
            with inject_logit_filter(self.logit_filter):
                result = self.model.transcribe(
                    tmp_path,
                    language=language,
                    fp16=(self.device.type == "cuda"),
                    verbose=False,
                )
            self._apply_technical_term_correction(result)
            return {
                "text": result.get("text", ""),
                "language": language,
            }
        finally:
            import os
            os.unlink(tmp_path)

    def transcribe_with_lid(self, audio_path: str,
                            language_segments: List[Dict],
                            sr: int = 16000) -> List[Dict]:
        """Transcribe using LID segments to set per-segment language.
        
        Args:
            audio_path: path to audio file
            language_segments: list of {"start": float, "end": float, "lang": str}
        """
        waveform, sr = load_audio(audio_path, sr=sr)
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        transcripts = []

        merged = self._merge_short_segments(language_segments, min_duration=1.0)

        for seg in merged:
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            chunk = waveform[start_sample:end_sample]

            if chunk.shape[-1] < sr * 0.5:
                continue

            result = self.transcribe_segment(chunk, sr, language=seg["lang"])
            result["start"] = seg["start"]
            result["end"] = seg["end"]
            transcripts.append(result)

        return transcripts

    def transcribe_full(self, audio_path: str, language: str = "en") -> Dict:
        """Transcribe full audio with a single language setting and logit bias."""
        print(f"[ASR] Transcribing {audio_path} (language={language})...")
        with inject_logit_filter(self.logit_filter):
            result = self.model.transcribe(
                audio_path,
                language=language,
                fp16=(self.device.type == "cuda"),
                verbose=False,
            )

        self._apply_technical_term_correction(result)
        return result

    def _apply_technical_term_correction(self, result: Dict):
        """Post-processing: correct common misrecognitions of technical terms."""
        corrections = {
            "spectrum": ["spectrem", "spectram"],
            "cepstrum": ["sepstrum", "kepstrum", "cestrum"],
            "mfcc": ["mfsc", "mfec"],
            "spectrogram": ["spectogram", "spectrgram"],
            "stochastic": ["stocastic", "stochastik"],
            "fourier": ["forier", "furier", "forrier"],
            "gaussian": ["gausian", "gausean"],
            "viterbi": ["viterby", "viterbe"],
            "markov": ["marcov", "markoff"],
            "phoneme": ["foneme", "phonem"],
            "formant": ["formant", "forment"],
            "cepstral": ["sepstral", "cestral"],
        }

        text = result.get("text", "")
        for correct, variants in corrections.items():
            for variant in variants:
                text = re.sub(re.escape(variant), correct, text, flags=re.IGNORECASE)
        result["text"] = text

    def _merge_short_segments(self, segments: List[Dict],
                               min_duration: float = 1.0) -> List[Dict]:
        """Merge segments shorter than min_duration with neighbors."""
        if not segments:
            return segments

        merged = [segments[0].copy()]
        for seg in segments[1:]:
            prev = merged[-1]
            if seg["lang"] == prev["lang"] or (seg["end"] - seg["start"]) < min_duration:
                prev["end"] = seg["end"]
            else:
                merged.append(seg.copy())

        return merged


def run_asr(audio_path: str, language_segments: Optional[List[Dict]] = None) -> List[Dict]:
    """Run constrained ASR on audio file."""
    asr = ConstrainedASR()

    if language_segments:
        transcripts = asr.transcribe_with_lid(audio_path, language_segments)
    else:
        result = asr.transcribe_full(audio_path)
        transcripts = [{
            "text": result["text"],
            "language": result.get("language", "en"),
            "start": 0.0,
            "end": 600.0,
        }]

    full_text = " ".join(t["text"] for t in transcripts)
    print(f"[ASR] Transcription ({len(full_text)} chars): {full_text[:200]}...")
    return transcripts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="denoised_segment.wav")
    args = parser.parse_args()
    run_asr(args.input)
