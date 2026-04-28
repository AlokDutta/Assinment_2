"""Task 3.1: Voice Embedding Extraction.

Extracts high-dimensional speaker embeddings (x-vector and d-vector)
from the student's 60-second reference recording.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import load_audio, get_device


class SpeakerEmbeddingExtractor:
    """Extracts x-vector (ECAPA-TDNN) and d-vector (Resemblyzer) speaker embeddings."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self._xvector_model = None
        self._dvector_encoder = None

    def _load_xvector_model(self):
        """Load SpeechBrain ECAPA-TDNN for x-vector extraction."""
        if self._xvector_model is None:
            from speechbrain.inference.speaker import EncoderClassifier
            self._xvector_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa",
                run_opts={"device": str(self.device)},
            )
        return self._xvector_model

    def _load_dvector_encoder(self):
        """Load Resemblyzer voice encoder for d-vector extraction."""
        if self._dvector_encoder is None:
            from resemblyzer import VoiceEncoder
            self._dvector_encoder = VoiceEncoder(device=str(self.device))
        return self._dvector_encoder

    def extract_xvector(self, audio_path: str) -> np.ndarray:
        """Extract x-vector (192-dim) using SpeechBrain ECAPA-TDNN.
        
        Returns:
            numpy array of shape (192,)
        """
        print("[Speaker Embed] Extracting x-vector...")
        model = self._load_xvector_model()
        waveform, sr = load_audio(audio_path, sr=16000)
        waveform = waveform.to(self.device)

        with torch.no_grad():
            embedding = model.encode_batch(waveform)

        xvec = embedding.squeeze().cpu().numpy()
        print(f"[Speaker Embed] x-vector shape: {xvec.shape}, norm: {np.linalg.norm(xvec):.4f}")
        return xvec

    def extract_dvector(self, audio_path: str) -> np.ndarray:
        """Extract d-vector (256-dim) using Resemblyzer.
        
        Returns:
            numpy array of shape (256,)
        """
        print("[Speaker Embed] Extracting d-vector...")
        encoder = self._load_dvector_encoder()

        from resemblyzer import preprocess_wav
        wav = preprocess_wav(audio_path)
        dvec = encoder.embed_utterance(wav)

        print(f"[Speaker Embed] d-vector shape: {dvec.shape}, norm: {np.linalg.norm(dvec):.4f}")
        return dvec

    def extract_all(self, audio_path: str, save_dir: str = "outputs") -> Dict[str, np.ndarray]:
        """Extract both x-vector and d-vector, save to disk."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        xvec = self.extract_xvector(audio_path)
        dvec = self.extract_dvector(audio_path)

        xvec_path = str(Path(save_dir) / "xvector.npy")
        dvec_path = str(Path(save_dir) / "dvector.npy")
        np.save(xvec_path, xvec)
        np.save(dvec_path, dvec)
        print(f"[Speaker Embed] Saved x-vector to {xvec_path}")
        print(f"[Speaker Embed] Saved d-vector to {dvec_path}")

        cosine_sim = np.dot(xvec[:min(len(xvec), len(dvec))],
                           dvec[:min(len(xvec), len(dvec))]) / (
                               np.linalg.norm(xvec[:min(len(xvec), len(dvec))]) *
                               np.linalg.norm(dvec[:min(len(xvec), len(dvec))]))
        print(f"[Speaker Embed] x-vector/d-vector cosine similarity "
              f"(first {min(len(xvec), len(dvec))} dims): {cosine_sim:.4f}")

        return {"xvector": xvec, "dvector": dvec}


def extract_speaker_embedding(audio_path: str, save_dir: str = "outputs") -> Dict[str, np.ndarray]:
    """Extract speaker embeddings from reference audio."""
    extractor = SpeakerEmbeddingExtractor()
    return extractor.extract_all(audio_path, save_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="student_voice_ref.wav")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    extract_speaker_embedding(args.input, args.output_dir)
