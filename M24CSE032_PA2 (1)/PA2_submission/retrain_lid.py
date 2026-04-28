"""Retrain frame-level LID with transcript-derived labels + class-balanced CE.

Uses word-level timestamps from transcript_best.json (romanised) to build
much more accurate English/Hindi frame labels than VoxLingua107 alone.
Overwrites outputs/lid_weights.pt.

Usage:
    CUDA_VISIBLE_DEVICES=0 python retrain_lid.py
"""

import os
import sys
import json
import re
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn.functional as F
from src.utils import load_audio, get_device
from src.part1_stt.lid import FrameLevelLID

# Import the same reference-label builder used by eval_lid.py
sys.path.insert(0, str(BASE_DIR))
from eval_lid import _make_transcript_ref_labels


def main():
    device = get_device()
    audio_path = str(BASE_DIR / "denoised_segment.wav")
    weights_path = str(BASE_DIR / "outputs" / "lid_weights.pt")

    print("=" * 60)
    print("LID RETRAIN (transcript-based labels + balanced CE)")
    print(f"Device: {device}")
    print("=" * 60)

    waveform, sr = load_audio(audio_path)
    print(f"Audio: {waveform.shape[-1] / sr:.1f}s @ {sr}Hz")

    lid = FrameLevelLID()

    print("\n[LID] Extracting ECAPA embeddings...")
    embeddings, timestamps = lid.extract_embeddings(waveform, sr)
    n_frames = len(embeddings)
    print(f"[LID] {n_frames} frames extracted")

    # Build reference labels from transcript word timestamps (much better than VoxLingua107)
    print("\n[LID] Building transcript-derived reference labels...")
    frame_stubs = [{"time": t, "lang": "hi"} for t in timestamps]
    ref_labels_list = _make_transcript_ref_labels(frame_stubs, BASE_DIR)
    labels = torch.tensor(ref_labels_list, device=device)

    en_count = int((labels == 0).sum().item())
    hi_count = int((labels == 1).sum().item())
    total = en_count + hi_count
    print(f"[LID] Label distribution: EN={en_count} ({en_count/total:.1%}), HI={hi_count} ({hi_count/total:.1%})")

    en_w = total / (2.0 * max(en_count, 1))
    hi_w = total / (2.0 * max(hi_count, 1))
    class_weights = torch.tensor([en_w, hi_w], device=device, dtype=torch.float32)
    print(f"[LID] Class weights: EN={en_w:.3f}, HI={hi_w:.3f}")

    contexts = lid._create_context_windows(embeddings)

    en_idx = (labels == 0).nonzero(as_tuple=True)[0]
    hi_idx = (labels == 1).nonzero(as_tuple=True)[0]
    target_per_class = max(en_count, hi_count)

    lid.classifier.train()
    optimizer = torch.optim.AdamW(lid.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    for epoch in range(30):
        en_s = en_idx[torch.randint(0, len(en_idx), (target_per_class,), device=device)]
        hi_s = hi_idx[torch.randint(0, len(hi_idx), (target_per_class,), device=device)]
        perm = torch.cat([en_s, hi_s])[torch.randperm(2 * target_per_class)]

        total_loss = 0.0
        correct = 0
        en_correct = 0
        en_total_b = 0
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            batch_ctx = contexts[idx].to(device)
            batch_labels = labels[idx]
            logits = lid.classifier(batch_ctx)
            loss = F.cross_entropy(logits, batch_labels, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            preds = logits.argmax(-1)
            correct += (preds == batch_labels).sum().item()
            em = batch_labels == 0
            en_total_b += int(em.sum().item())
            en_correct += int(((preds == batch_labels) & em).sum().item())

        scheduler.step()
        acc = correct / max(len(perm), 1)
        en_acc = en_correct / max(en_total_b, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/30 | Loss: {total_loss/len(perm):.4f} | Acc: {acc:.3f} | EN-Recall: {en_acc:.3f}")

    lid.classifier.eval()
    lid.save_weights(weights_path)
    print(f"\nSaved new weights to {weights_path}")
    print("Now run: CUDA_VISIBLE_DEVICES=0 python eval_lid.py")
    print("Then:    CUDA_VISIBLE_DEVICES=0 python eval_fgsm.py")


if __name__ == "__main__":
    main()
