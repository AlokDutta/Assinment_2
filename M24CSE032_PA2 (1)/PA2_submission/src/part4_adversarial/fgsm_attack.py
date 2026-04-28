"""Task 4.2: Adversarial Noise Injection using FGSM.

Implements Fast Gradient Sign Method to find the minimum perturbation
that causes the LID system to misclassify Hindi speech as English,
while maintaining SNR > 40 dB (inaudible to humans).

Uses differentiable forward pass through the actual ECAPA-TDNN + 
Multi-Head Attention LID classifier for proper gradient computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import load_audio, save_audio, compute_snr, get_device


class FGSMAttacker:
    """Fast Gradient Sign Method adversarial attack on the LID system.
    
    Uses the real ECAPA-TDNN + Multi-Head Attention classifier with
    differentiable forward pass (bypassing torch.no_grad wrapper)
    for proper gradient-based adversarial perturbation.
    
    Finds minimum epsilon such that:
    1. LID misclassifies Hindi as English
    2. SNR > 40 dB (perturbation is inaudible)
    """

    def __init__(self, lid_system, device: Optional[torch.device] = None):
        self.lid = lid_system
        self.device = device or get_device()

    def _differentiable_lid_forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run frame-level LID forward pass WITH gradient tracking.
        
        Processes 400ms windows with 100ms hop (matching the real predict
        pipeline) through ECAPA-TDNN, then runs the Multi-Head Attention
        classifier on context windows.
        """
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        sr = 16000
        window_samples = int(self.lid.window_ms / 1000 * sr)
        hop_samples = int(self.lid.hop_ms / 1000 * sr)
        total = waveform.shape[-1]

        embeddings = []
        ones_lens = torch.ones(1, device=self.device)
        for start in range(0, total - window_samples + 1, hop_samples):
            chunk = waveform[start:start + window_samples].unsqueeze(0)
            mel_fbanks = self.lid.ecapa.mods.compute_features(chunk)
            # Include utterance-level feature normalisation — skipping this
            # breaks the gradient path because the embedding_model expects
            # normalised features (matches SpeechBrain's encode_batch flow).
            if hasattr(self.lid.ecapa.mods, "mean_var_norm"):
                mel_fbanks = self.lid.ecapa.mods.mean_var_norm(mel_fbanks, ones_lens)
            emb = self.lid.ecapa.mods.embedding_model(mel_fbanks)
            embeddings.append(emb.squeeze())

        if not embeddings:
            return torch.zeros(1, 2, device=self.device)

        embeddings = torch.stack(embeddings)

        ctx_size = self.lid.classifier.context_size
        contexts = []
        for i in range(len(embeddings)):
            ctx_indices = []
            for offset in range(-(ctx_size // 2), ctx_size // 2 + 1):
                idx = max(0, min(len(embeddings) - 1, i + offset))
                ctx_indices.append(idx)
            ctx = embeddings[ctx_indices]
            contexts.append(ctx)

        contexts = torch.stack(contexts)
        logits = self.lid.classifier(contexts)
        return logits

    def fgsm_attack(self, waveform: torch.Tensor, epsilon: float,
                    target_class: int = 0, n_iter: int = 10,
                    early_stop: bool = True) -> torch.Tensor:
        """Iterative FGSM (I-FGSM/PGD) attack using the real LID pipeline.

        Each iteration:
        1. Compute mel features → ECAPA embedding → classifier logits (differentiable)
        2. Compute cross-entropy loss toward target class
        3. Backprop to input waveform
        4. Apply sign gradient perturbation within epsilon-ball

        If early_stop, returns as soon as >=50% frames flip to target, using
        the minimal accumulated perturbation (preserves SNR budget).
        """
        orig = waveform.to(self.device).clone().detach()
        adv = orig.clone()
        step_size = epsilon / max(n_iter, 1)

        self.lid.classifier.eval()
        for p in self.lid.classifier.parameters():
            p.requires_grad_(False)

        grad_ok_logged = False
        for it in range(n_iter):
            adv = adv.clone().detach().requires_grad_(True)

            logits = self._differentiable_lid_forward(adv.squeeze())
            target = torch.full((logits.shape[0],), target_class,
                               dtype=torch.long, device=self.device)
            loss = F.cross_entropy(logits, target)
            loss.backward()

            if adv.grad is not None and adv.grad.abs().max().item() > 1e-12:
                if not grad_ok_logged:
                    print(f"    [FGSM] grad OK at iter {it+1}, max={adv.grad.abs().max().item():.2e}")
                    grad_ok_logged = True
                adv = adv - step_size * adv.grad.sign()
                perturbation = torch.clamp(adv - orig, -epsilon, epsilon)
                adv = torch.clamp(orig + perturbation, -1.0, 1.0)
            else:
                if not grad_ok_logged:
                    print(f"    [FGSM] WARNING: zero/None gradient at iter {it+1}, using random sign")
                    grad_ok_logged = True
                noise = torch.sign(torch.randn_like(orig)) * step_size
                perturbation = torch.clamp((adv - orig) + noise, -epsilon, epsilon)
                adv = torch.clamp(orig + perturbation, -1.0, 1.0)

            if early_stop and (it + 1) >= 2:
                with torch.no_grad():
                    check_logits = self._differentiable_lid_forward(adv.squeeze())
                    preds = check_logits.argmax(-1)
                    if (preds == target_class).float().mean().item() >= 0.5:
                        break

        return adv.detach()

    def find_minimum_epsilon(self, waveform: torch.Tensor,
                             full_audio_path: str,
                             min_snr: float = 40.0,
                             target_class: int = 0,
                             max_iterations: int = 20) -> Dict:
        """Binary search for minimum epsilon that flips LID prediction.
        
        Constraints:
        - LID must predict target_class (English) instead of Hindi
        - SNR must be > min_snr dB
        """
        print("[FGSM] Searching for minimum epsilon...")

        lo, hi = 0.0, 0.20
        best_epsilon = None
        best_result = None
        min_flip_epsilon = None
        min_flip_snr = None

        for iteration in range(max_iterations):
            epsilon = (lo + hi) / 2
            adv_waveform = self.fgsm_attack(waveform, epsilon, target_class,
                                            n_iter=80, early_stop=True)

            snr = compute_snr(waveform.squeeze().cpu(), adv_waveform.squeeze().cpu())

            results = self.lid.predict(adv_waveform.squeeze(0).cpu(), sr=16000)
            if results:
                pred_counts = {"en": 0, "hi": 0}
                for r in results:
                    pred_counts[r["lang"]] += 1
                dominant = max(pred_counts, key=pred_counts.get)
                flipped = (dominant == "en")
            else:
                flipped = False

            snr_str = f"{snr:.1f}" if snr != float('inf') else "inf"
            print(f"  Iter {iteration+1}: eps={epsilon:.6f}, SNR={snr_str}dB, "
                  f"flipped={flipped}")

            if flipped:
                if min_flip_epsilon is None or epsilon < min_flip_epsilon:
                    min_flip_epsilon = epsilon
                    min_flip_snr = snr

                if snr > min_snr:
                    best_epsilon = epsilon
                    best_result = {
                        "epsilon": epsilon,
                        "snr": snr,
                        "flipped": True,
                        "predictions": results,
                    }
                    hi = epsilon
                else:
                    hi = epsilon
            else:
                lo = epsilon

        if best_result is None:
            if min_flip_epsilon is not None:
                print(f"[FGSM] Flip achieved at eps={min_flip_epsilon:.6f} "
                      f"(SNR={min_flip_snr:.1f}dB) but below {min_snr}dB constraint.")
                best_result = {
                    "epsilon": min_flip_epsilon,
                    "snr": min_flip_snr,
                    "flipped": True,
                    "snr_constraint_met": False,
                    "predictions": [],
                }
            else:
                print("[FGSM] Could not flip LID predictions.")
                adv = self.fgsm_attack(waveform, hi, target_class, n_iter=20)
                best_result = {
                    "epsilon": hi,
                    "snr": compute_snr(waveform.squeeze().cpu(), adv.squeeze().cpu()),
                    "flipped": False,
                    "predictions": [],
                }

        return best_result

    def run_attack(self, audio_path: str, start_sec: float, duration_sec: float = 5.0,
                   output_dir: str = "outputs") -> Dict:
        """Run FGSM attack on a Hindi segment."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        waveform, sr = load_audio(audio_path, sr=16000)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        start_sample = int(start_sec * sr)
        end_sample = int((start_sec + duration_sec) * sr)
        segment = waveform[:, start_sample:end_sample]

        print(f"[FGSM] Attacking {duration_sec}s Hindi segment starting at {start_sec}s")

        original_results = self.lid.predict(segment.squeeze(0), sr=16000)
        print(f"[FGSM] Original prediction distribution:")
        if original_results:
            pred_counts = {"en": 0, "hi": 0}
            for r in original_results:
                pred_counts[r["lang"]] += 1
            print(f"  English: {pred_counts['en']}, Hindi: {pred_counts['hi']}")

        result = self.find_minimum_epsilon(segment, audio_path, min_snr=40.0, target_class=0)

        if result.get("epsilon") is not None:
            adv_waveform = self.fgsm_attack(segment, result["epsilon"], target_class=0)
            adv_path = str(Path(output_dir) / "adversarial_segment.wav")
            save_audio(adv_waveform, adv_path, sr)
            result["adversarial_audio_path"] = adv_path

        result["original_predictions"] = original_results
        result["segment_start"] = start_sec
        result["segment_duration"] = duration_sec

        print(f"\n[FGSM] Results:")
        eps_val = result.get('epsilon')
        snr_val = result.get('snr')
        print(f"  Minimum epsilon: {eps_val:.6f}" if eps_val else "  Minimum epsilon: N/A")
        print(f"  SNR: {snr_val:.1f} dB" if snr_val and snr_val != float('inf') else f"  SNR: {snr_val}")
        print(f"  LID flipped: {result['flipped']}")

        return result


def run_fgsm_attack(audio_path: str, lid_system, hindi_start: float = 0.0,
                    output_dir: str = "outputs") -> Dict:
    """Run FGSM attack on a Hindi segment identified by LID."""
    attacker = FGSMAttacker(lid_system)
    return attacker.run_attack(audio_path, hindi_start, duration_sec=5.0,
                               output_dir=output_dir)
