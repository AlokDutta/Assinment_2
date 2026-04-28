"""Task 3.3: Zero-Shot Cross-Lingual Voice Cloning with XTTS v2.

XTTS v2 is a VITS-family architecture (extends VITS with GPT-2 latent
conditioning and HiFi-GAN decoder). Synthesizes Meitei text (transliterated
to Devanagari) in Hindi mode using zero-shot voice cloning from the
student's 60-second reference recording. Output at 24 kHz.
"""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_device


class XTTSv2Synthesizer:
    """XTTS v2 synthesizer with zero-shot voice cloning."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.model = None
        self.output_sr = 24000

    def load_model(self):
        """Load XTTS v2 model via Coqui TTS."""
        if self.model is not None:
            return

        print("[TTS] Loading XTTS v2 model...")
        from TTS.api import TTS
        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
            str(self.device)
        )
        print("[TTS] XTTS v2 loaded successfully")

    def synthesize_sentence(self, text: str, speaker_wav: str,
                            language: str = "hi") -> np.ndarray:
        """Synthesize a single sentence with zero-shot voice cloning.
        
        Args:
            text: Input text (Devanagari for Hindi mode)
            speaker_wav: Path to speaker reference audio
            language: Language code ('hi' for Hindi mode)
        Returns:
            Audio waveform as numpy array at 24 kHz
        """
        self.load_model()

        if not text.strip():
            return np.zeros(int(0.5 * self.output_sr))

        try:
            wav = self.model.tts(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
            )
            return np.array(wav, dtype=np.float32)
        except Exception as e:
            print(f"[TTS] Warning: synthesis failed for '{text[:50]}...': {e}")
            return np.zeros(int(0.5 * self.output_sr))

    def synthesize_transcript(self, translated_segments: List[Dict],
                              speaker_wav: str,
                              output_path: str,
                              language: str = "hi") -> str:
        """Synthesize full translated transcript.
        
        Args:
            translated_segments: List of dicts with 'devanagari' key for text
            speaker_wav: Path to speaker reference audio
            output_path: Output WAV file path
            language: TTS language code
        Returns:
            Path to output file
        """
        self.load_model()
        
        all_audio = []
        silence = np.zeros(int(0.3 * self.output_sr))

        total = len(translated_segments)
        for i, seg in enumerate(translated_segments):
            text = seg.get("devanagari", seg.get("mni_latin", ""))
            if not text.strip():
                all_audio.append(silence)
                continue

            sentences = self._split_into_sentences(text)

            for sent in sentences:
                if not sent.strip():
                    continue
                print(f"[TTS] Synthesizing ({i+1}/{total}): {sent[:60]}...")
                wav = self.synthesize_sentence(sent, speaker_wav, language)
                all_audio.append(wav)
                all_audio.append(silence)

        if not all_audio:
            all_audio = [np.zeros(int(1.0 * self.output_sr))]

        full_audio = np.concatenate(all_audio)

        peak = np.max(np.abs(full_audio))
        if peak > 0:
            full_audio = full_audio / peak * 0.95

        sf.write(output_path, full_audio, self.output_sr)
        duration = len(full_audio) / self.output_sr
        print(f"[TTS] Saved synthesized audio: {output_path} "
              f"({duration:.1f}s, {self.output_sr} Hz)")

        return output_path

    def _split_into_sentences(self, text: str, max_chars: int = 200) -> List[str]:
        """Split text into sentences suitable for TTS processing."""
        delimiters = ['।', '.', '!', '?', ',', ';']
        sentences = [text]
        for delim in delimiters:
            new_sentences = []
            for sent in sentences:
                parts = sent.split(delim)
                for j, part in enumerate(parts):
                    part = part.strip()
                    if part:
                        if j < len(parts) - 1:
                            new_sentences.append(part + delim)
                        else:
                            new_sentences.append(part)
            sentences = new_sentences

        final = []
        for sent in sentences:
            if len(sent) > max_chars:
                words = sent.split()
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 > max_chars:
                        if current:
                            final.append(current.strip())
                        current = word
                    else:
                        current = current + " " + word if current else word
                if current:
                    final.append(current.strip())
            else:
                final.append(sent)

        return [s for s in final if s.strip()]

    def get_internal_speaker_embedding(self, speaker_wav: str) -> np.ndarray:
        """Extract XTTS v2's internal speaker embedding for comparison with x-vector."""
        self.load_model()
        try:
            gpt_cond_latent, speaker_embedding = self.model.synthesizer.tts_model.get_conditioning_latents(
                audio_path=[speaker_wav]
            )
            return speaker_embedding.squeeze().cpu().numpy()
        except Exception as e:
            print(f"[TTS] Could not extract internal embedding: {e}")
            return np.zeros(512)


def synthesize_lecture(translated_segments: List[Dict],
                       speaker_wav: str,
                       output_path: str = "output_LRL_cloned.wav") -> str:
    """Synthesize the full lecture in Meitei using XTTS v2."""
    synth = XTTSv2Synthesizer()
    return synth.synthesize_transcript(
        translated_segments, speaker_wav, output_path, language="hi"
    )


if __name__ == "__main__":
    test_segments = [
        {"devanagari": "यह एक परीक्षण वाक्य है", "mni_latin": "masi amuk test wahai ni"},
    ]
    synthesize_lecture(test_segments, "student_voice_ref.wav", "test_output.wav")
