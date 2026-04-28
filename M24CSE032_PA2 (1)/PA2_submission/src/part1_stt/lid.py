"""Task 1.1: Multi-Head Language Identification (LID).

Frame-level LID using SpeechBrain ECAPA-TDNN embeddings + Multi-Head Self-Attention
classifier to distinguish English vs Hindi in code-switched speech.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import load_audio, get_device


class MultiHeadLIDClassifier(nn.Module):
    """Multi-Head Self-Attention classifier for frame-level language identification.
    
    Takes a sequence of ECAPA-TDNN embeddings (context window) and predicts
    the language (English=0, Hindi=1) for the center frame.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4,
                 context_size: int = 5, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.context_size = context_size
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, context_size, embed_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, context_size, embed_dim) sequence of embeddings
        Returns:
            logits: (batch, num_classes)
        """
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.shape[1], :]

        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        center = x[:, x.shape[1] // 2, :]
        logits = self.classifier(center)
        return logits


class FrameLevelLID:
    """Frame-level Language Identification system.
    
    Uses SpeechBrain ECAPA-TDNN for embeddings and Multi-Head Attention for classification.
    """

    LANG_MAP = {0: "en", 1: "hi"}
    LANG_MAP_INV = {"en": 0, "hi": 1}

    def __init__(self, window_ms: int = 400, hop_ms: int = 100,
                 context_size: int = 5, device: Optional[torch.device] = None):
        self.window_ms = window_ms
        self.hop_ms = hop_ms
        self.context_size = context_size
        self.device = device or get_device()
        self.sr = 16000

        self._load_ecapa_model()
        self.embed_dim = 256
        self.classifier = MultiHeadLIDClassifier(
            embed_dim=self.embed_dim, num_heads=4,
            context_size=context_size
        ).to(self.device)

    def _load_ecapa_model(self):
        """Load pre-trained SpeechBrain ECAPA-TDNN for language identification."""
        from speechbrain.inference.classifiers import EncoderClassifier
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="pretrained_models/lang-id-ecapa",
            run_opts={"device": str(self.device)},
        )

    def extract_embeddings(self, waveform: torch.Tensor, sr: int = 16000) -> Tuple[torch.Tensor, List[float]]:
        """Extract frame-level ECAPA embeddings with sliding window.
        
        Returns:
            embeddings: (n_frames, embed_dim) tensor
            timestamps: list of center timestamps in seconds
        """
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        window_samples = int(self.window_ms / 1000 * sr)
        hop_samples = int(self.hop_ms / 1000 * sr)
        total_samples = waveform.shape[-1]

        embeddings = []
        timestamps = []

        for start in range(0, total_samples - window_samples + 1, hop_samples):
            end = start + window_samples
            chunk = waveform[start:end].unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.ecapa.encode_batch(chunk)
            embeddings.append(emb.squeeze(0).squeeze(0))
            center_time = (start + window_samples / 2) / sr
            timestamps.append(center_time)

        if not embeddings:
            return torch.zeros(0, self.embed_dim), []

        embeddings = torch.stack(embeddings)
        return embeddings, timestamps

    def _create_context_windows(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Create context windows for the attention classifier."""
        n_frames = embeddings.shape[0]
        pad = self.context_size // 2
        padded = F.pad(embeddings.unsqueeze(0).transpose(1, 2),
                       (pad, pad), mode='replicate').transpose(1, 2).squeeze(0)

        contexts = []
        for i in range(n_frames):
            ctx = padded[i:i + self.context_size]
            contexts.append(ctx)
        return torch.stack(contexts)

    def predict(self, waveform: torch.Tensor, sr: int = 16000) -> List[Dict]:
        """Run full LID prediction on waveform.
        
        Returns list of dicts: [{"time": float, "lang": str, "confidence": float}, ...]
        """
        embeddings, timestamps = self.extract_embeddings(waveform, sr)
        if len(embeddings) == 0:
            return []

        contexts = self._create_context_windows(embeddings)

        self.classifier.eval()
        batch_size = 64
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(contexts), batch_size):
                batch = contexts[i:i + batch_size].to(self.device)
                logits = self.classifier(batch)
                all_logits.append(logits)

        all_logits = torch.cat(all_logits, dim=0)
        probs = F.softmax(all_logits, dim=-1)
        predictions = probs.argmax(dim=-1)
        confidences = probs.max(dim=-1).values

        predictions = self._median_filter(predictions, kernel_size=5)

        results = []
        for i, (t, pred, conf) in enumerate(zip(timestamps, predictions, confidences)):
            results.append({
                "time": t,
                "lang": self.LANG_MAP[pred.item()],
                "confidence": conf.item(),
            })

        return results

    def _median_filter(self, predictions: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Apply median filter to smooth predictions and reduce flickering."""
        if len(predictions) < kernel_size:
            return predictions
        padded = F.pad(predictions.float().unsqueeze(0).unsqueeze(0),
                       (kernel_size // 2, kernel_size // 2), mode='replicate')
        filtered = padded.unfold(-1, kernel_size, 1).median(dim=-1).values
        return filtered.squeeze().long()

    def get_language_segments(self, results: List[Dict]) -> List[Dict]:
        """Convert frame-level predictions to segment-level.
        
        Returns: [{"start": float, "end": float, "lang": str}, ...]
        """
        if not results:
            return []

        segments = []
        current_lang = results[0]["lang"]
        start_time = results[0]["time"]

        for i in range(1, len(results)):
            if results[i]["lang"] != current_lang:
                segments.append({
                    "start": start_time,
                    "end": results[i - 1]["time"],
                    "lang": current_lang,
                })
                current_lang = results[i]["lang"]
                start_time = results[i]["time"]

        segments.append({
            "start": start_time,
            "end": results[-1]["time"],
            "lang": current_lang,
        })

        return segments

    def train_on_pseudolabels(self, waveform: torch.Tensor, sr: int = 16000,
                               epochs: int = 30, lr: float = 1e-3):
        """Train the classifier using VoxLingua107 pseudo-labels."""
        print("[LID] Extracting embeddings for training...")
        embeddings, timestamps = self.extract_embeddings(waveform, sr)
        if len(embeddings) == 0:
            print("[LID] No embeddings extracted, skipping training.")
            return

        print("[LID] Generating pseudo-labels from VoxLingua107...")
        window_samples = int(self.window_ms / 1000 * sr)
        hop_samples = int(self.hop_ms / 1000 * sr)
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        pseudo_labels = []
        for start in range(0, waveform.shape[-1] - window_samples + 1, hop_samples):
            end = start + window_samples
            chunk = waveform[start:end].unsqueeze(0).to(self.device)
            with torch.no_grad():
                out_prob, score, index, text_lab = self.ecapa.classify_batch(chunk)
            lang = text_lab[0]
            label = 0 if lang in ["en: English", "English"] or "en" in str(lang).lower()[:3] else 1
            pseudo_labels.append(label)

        if len(pseudo_labels) != len(embeddings):
            pseudo_labels = pseudo_labels[:len(embeddings)]

        labels = torch.tensor(pseudo_labels, device=self.device)
        contexts = self._create_context_windows(embeddings)

        en_count = int((labels == 0).sum().item())
        hi_count = int((labels == 1).sum().item())
        total = en_count + hi_count
        print(f"[LID] Pseudo-label distribution: EN={en_count} ({en_count/total:.1%}), HI={hi_count} ({hi_count/total:.1%})")

        en_w = total / (2.0 * max(en_count, 1))
        hi_w = total / (2.0 * max(hi_count, 1))
        class_weights = torch.tensor([en_w, hi_w], device=self.device, dtype=torch.float32)
        print(f"[LID] Class weights: EN={en_w:.3f}, HI={hi_w:.3f}")

        en_idx = (labels == 0).nonzero(as_tuple=True)[0]
        hi_idx = (labels == 1).nonzero(as_tuple=True)[0]
        target_per_class = max(en_count, hi_count)

        self.classifier.train()
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            en_sample = en_idx[torch.randint(0, len(en_idx), (target_per_class,), device=self.device)] if len(en_idx) > 0 else en_idx
            hi_sample = hi_idx[torch.randint(0, len(hi_idx), (target_per_class,), device=self.device)] if len(hi_idx) > 0 else hi_idx
            balanced_idx = torch.cat([en_sample, hi_sample])
            perm = balanced_idx[torch.randperm(len(balanced_idx))]

            total_loss = 0.0
            correct = 0
            en_correct = 0
            en_total = 0
            for i in range(0, len(perm), 64):
                idx = perm[i:i + 64]
                batch_ctx = contexts[idx].to(self.device)
                batch_labels = labels[idx]

                logits = self.classifier(batch_ctx)
                loss = F.cross_entropy(logits, batch_labels, weight=class_weights)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(idx)
                preds = logits.argmax(-1)
                correct += (preds == batch_labels).sum().item()
                en_mask = batch_labels == 0
                en_total += int(en_mask.sum().item())
                en_correct += int(((preds == batch_labels) & en_mask).sum().item())

            scheduler.step()
            acc = correct / max(len(perm), 1)
            en_acc = en_correct / max(en_total, 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  [LID Train] Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(perm):.4f} | Acc: {acc:.3f} | EN-Recall: {en_acc:.3f}")

        self.classifier.eval()
        print("[LID] Training complete.")

    def save_weights(self, path: str):
        torch.save(self.classifier.state_dict(), path)
        print(f"[LID] Saved classifier weights to {path}")

    def load_weights(self, path: str):
        self.classifier.load_state_dict(torch.load(path, map_location=self.device))
        self.classifier.eval()
        print(f"[LID] Loaded classifier weights from {path}")


def run_lid(audio_path: str, output_dir: str = "outputs") -> Tuple[List[Dict], List[Dict]]:
    """Run LID on an audio file. Returns (frame_results, segments)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    waveform, sr = load_audio(audio_path)

    lid = FrameLevelLID()
    lid.train_on_pseudolabels(waveform, sr)

    frame_results = lid.predict(waveform, sr)
    segments = lid.get_language_segments(frame_results)

    lid.save_weights(str(Path(output_dir) / "lid_weights.pt"))

    print(f"[LID] Found {len(segments)} language segments")
    for seg in segments[:10]:
        print(f"  {seg['start']:.2f}s - {seg['end']:.2f}s : {seg['lang']}")

    return frame_results, segments


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="denoised_segment.wav")
    args = parser.parse_args()
    run_lid(args.input)
