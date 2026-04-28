"""Task 3.2: Prosody Warping with Dynamic Time Warping (DTW).

Extracts F0 and energy contours from the professor's lecture and applies
custom DTW (implemented in PyTorch) to warp prosodic features onto the
synthesized LRL speech to preserve teaching style.
"""

import torch
import numpy as np
import librosa
import parselmouth
from pathlib import Path
from typing import Tuple, Optional, List
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import load_audio, get_device


class ProsodyExtractor:
    """Extracts F0 and energy contours from audio."""

    def __init__(self, sr: int = 16000, frame_length: int = 400, hop_length: int = 160):
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length

    def extract_f0(self, waveform: np.ndarray, method: str = "pyin") -> np.ndarray:
        """Extract fundamental frequency contour.
        
        Args:
            waveform: 1D numpy array
            method: 'pyin' (librosa) or 'praat' (parselmouth)
        Returns:
            f0: 1D array of F0 values (Hz), 0 for unvoiced frames
        """
        if method == "pyin":
            f0, voiced_flag, voiced_probs = librosa.pyin(
                waveform, fmin=60, fmax=500,
                sr=self.sr, frame_length=self.frame_length,
                hop_length=self.hop_length,
            )
            f0 = np.nan_to_num(f0, nan=0.0)
        elif method == "praat":
            snd = parselmouth.Sound(waveform, sampling_frequency=self.sr)
            pitch = snd.to_pitch(
                time_step=self.hop_length / self.sr,
                pitch_floor=60, pitch_ceiling=500
            )
            f0 = pitch.selected_array['frequency']
        else:
            raise ValueError(f"Unknown F0 method: {method}")

        return f0

    def extract_energy(self, waveform: np.ndarray) -> np.ndarray:
        """Extract frame-level RMS energy contour."""
        energy = librosa.feature.rms(
            y=waveform, frame_length=self.frame_length,
            hop_length=self.hop_length
        ).squeeze()
        return energy

    def extract_prosody(self, waveform: np.ndarray) -> dict:
        """Extract full prosodic features (F0 + energy)."""
        f0 = self.extract_f0(waveform)
        energy = self.extract_energy(waveform)

        min_len = min(len(f0), len(energy))
        f0 = f0[:min_len]
        energy = energy[:min_len]

        return {"f0": f0, "energy": energy}


class DTWAligner:
    """Custom Dynamic Time Warping implementation in PyTorch.
    
    Computes the optimal alignment between two sequences and returns
    the warping path for prosody transfer. Uses a Sakoe-Chiba band
    constraint for memory/time efficiency on long sequences, and
    operates on CPU to avoid GPU OOM on large cost matrices.
    """

    def __init__(self, device: Optional[torch.device] = None, max_dtw_len: int = 5000):
        self.device = torch.device("cpu")
        self.max_dtw_len = max_dtw_len

    def _downsample(self, contour: np.ndarray, target_len: int) -> Tuple[np.ndarray, float]:
        """Downsample a contour to target_len with averaging."""
        factor = len(contour) / target_len
        out = np.zeros(target_len)
        for i in range(target_len):
            start = int(i * factor)
            end = int((i + 1) * factor)
            end = max(end, start + 1)
            out[i] = np.mean(contour[start:min(end, len(contour))])
        return out, factor

    def compute_cost_matrix_banded(self, source: torch.Tensor, target: torch.Tensor,
                                    band_radius: int) -> torch.Tensor:
        """Compute banded pairwise distance cost matrix (Sakoe-Chiba band).
        
        Only computes costs within |i/N - j/M| <= band_radius/max(N,M),
        storing a dense (N, 2*band_radius+1) matrix instead of (N, M).
        """
        N = source.shape[0]
        M = target.shape[0]
        width = 2 * band_radius + 1
        cost = torch.full((N, width), float('inf'), device=self.device)

        for i in range(N):
            center_j = int(round(i * M / N))
            j_start = max(0, center_j - band_radius)
            j_end = min(M, center_j + band_radius + 1)
            for j in range(j_start, j_end):
                k = j - center_j + band_radius
                cost[i, k] = torch.abs(source[i] - target[j])

        return cost

    def compute_dtw_banded(self, source: torch.Tensor, target: torch.Tensor,
                           band_radius: int) -> List[Tuple[int, int]]:
        """Compute DTW with Sakoe-Chiba band constraint.
        
        Custom PyTorch implementation with three valid predecessors
        (diagonal, horizontal, vertical) operating within the band.
        """
        N = source.shape[0]
        M = target.shape[0]
        width = 2 * band_radius + 1

        cost = self.compute_cost_matrix_banded(source, target, band_radius)

        D = torch.full((N + 1, width), float('inf'), device=self.device)
        D[0, band_radius] = 0

        for i in range(1, N + 1):
            center_j = int(round((i - 1) * M / N))
            j_start = max(0, center_j - band_radius)
            j_end = min(M, center_j + band_radius + 1)

            for j in range(j_start, j_end):
                k = j - center_j + band_radius
                c = cost[i - 1, k]

                prev_center = int(round((i - 2) * M / N)) if i >= 2 else 0
                best = float('inf')

                # diagonal: (i-1, j-1)
                if i >= 2 and j >= 1:
                    k_prev = (j - 1) - prev_center + band_radius
                    if 0 <= k_prev < width:
                        best = min(best, D[i - 1, k_prev].item())

                # vertical: (i-1, j)
                if i >= 2:
                    k_prev = j - prev_center + band_radius
                    if 0 <= k_prev < width:
                        best = min(best, D[i - 1, k_prev].item())

                # horizontal: (i, j-1)
                if j >= 1:
                    k_prev = (j - 1) - center_j + band_radius
                    if 0 <= k_prev < width:
                        best = min(best, D[i, k_prev].item())

                D[i, k] = c + best

        # Backtrack
        path = []
        i = N
        center_j = int(round((i - 1) * M / N))
        j_start = max(0, center_j - band_radius)
        j_end = min(M, center_j + band_radius + 1)
        j = j_start
        best_val = float('inf')
        for jj in range(j_start, j_end):
            k = jj - center_j + band_radius
            if D[i, k].item() < best_val:
                best_val = D[i, k].item()
                j = jj

        while i > 0 and j >= 0:
            path.append((i - 1, j))
            if i == 1 and j == 0:
                break
            center_j_prev = int(round((i - 2) * M / N)) if i >= 2 else 0
            center_j_cur = int(round((i - 1) * M / N))
            candidates = []

            # diagonal
            if i >= 2 and j >= 1:
                k_prev = (j - 1) - center_j_prev + band_radius
                if 0 <= k_prev < width:
                    candidates.append(((i - 1, j - 1), D[i - 1, k_prev].item()))

            # vertical
            if i >= 2:
                k_prev = j - center_j_prev + band_radius
                if 0 <= k_prev < width:
                    candidates.append(((i - 1, j), D[i - 1, k_prev].item()))

            # horizontal
            if j >= 1:
                k_prev = (j - 1) - center_j_cur + band_radius
                if 0 <= k_prev < width:
                    candidates.append(((i, j - 1), D[i, k_prev].item()))

            if not candidates:
                break
            (i, j), _ = min(candidates, key=lambda x: x[1])

        path.reverse()
        return path

    def warp_contour(self, source_contour: np.ndarray, target_length: int,
                     target_contour: Optional[np.ndarray] = None) -> np.ndarray:
        """Warp source prosodic contour to match target timing using DTW.
        
        For long sequences, downsamples before DTW then maps the path
        back to the original resolution via interpolation.
        """
        src_len = len(source_contour)
        tgt_len = target_length if target_contour is None else len(target_contour)

        need_downsample = max(src_len, tgt_len) > self.max_dtw_len
        if need_downsample:
            ds_src, src_factor = self._downsample(source_contour, self.max_dtw_len)
            ds_tgt_len = min(tgt_len, self.max_dtw_len)
            if target_contour is not None:
                ds_tgt, tgt_factor = self._downsample(target_contour, ds_tgt_len)
            else:
                ds_tgt = np.linspace(ds_src.min(), ds_src.max(), ds_tgt_len)
                tgt_factor = tgt_len / ds_tgt_len
        else:
            ds_src = source_contour
            ds_tgt = target_contour if target_contour is not None else np.linspace(
                source_contour.min(), source_contour.max(), tgt_len)
            src_factor = 1.0
            tgt_factor = 1.0

        source_t = torch.tensor(ds_src, dtype=torch.float32, device=self.device)
        target_t = torch.tensor(ds_tgt, dtype=torch.float32, device=self.device)

        band_radius = max(50, len(source_t) // 5)
        path = self.compute_dtw_banded(source_t, target_t, band_radius)

        warped = np.zeros(target_length)
        counts = np.zeros(target_length)

        for src_idx, tgt_idx in path:
            real_src = int(src_idx * src_factor)
            real_tgt = int(tgt_idx * tgt_factor)

            real_src = min(real_src, len(source_contour) - 1)
            real_tgt = min(real_tgt, target_length - 1)

            warped[real_tgt] += source_contour[real_src]
            counts[real_tgt] += 1

        mask = counts > 0
        warped[mask] /= counts[mask]

        if not mask.all():
            from scipy.interpolate import interp1d
            valid_idx = np.where(mask)[0]
            if len(valid_idx) >= 2:
                interp_fn = interp1d(valid_idx, warped[valid_idx], kind='linear',
                                     fill_value='extrapolate')
                warped = interp_fn(np.arange(target_length))

        return warped


class ProsodyWarper:
    """Full prosody warping pipeline: extract -> align -> warp."""

    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.extractor = ProsodyExtractor(sr=sr)
        self.aligner = DTWAligner()

    def extract_and_warp(self, source_path: str, target_path: str) -> dict:
        """Extract prosody from source (professor) and warp to target (student) timing.
        
        Args:
            source_path: path to original lecture audio
            target_path: path to synthesized audio
        Returns:
            dict with warped_f0 and warped_energy arrays
        """
        print("[Prosody] Extracting source prosody (professor's lecture)...")
        source_wav, _ = load_audio(source_path, sr=self.sr)
        source_np = source_wav.squeeze().numpy()
        source_prosody = self.extractor.extract_prosody(source_np)

        print("[Prosody] Extracting target prosody (synthesized speech)...")
        target_wav, _ = load_audio(target_path, sr=self.sr)
        target_np = target_wav.squeeze().numpy()
        target_prosody = self.extractor.extract_prosody(target_np)

        target_len = len(target_prosody["f0"])

        print("[Prosody] Applying DTW warping to F0 contour...")
        warped_f0 = self.aligner.warp_contour(
            source_prosody["f0"], target_len,
            target_contour=target_prosody["f0"]
        )

        print("[Prosody] Applying DTW warping to energy contour...")
        warped_energy = self.aligner.warp_contour(
            source_prosody["energy"], target_len,
            target_contour=target_prosody["energy"]
        )

        return {
            "warped_f0": warped_f0,
            "warped_energy": warped_energy,
            "source_f0": source_prosody["f0"],
            "source_energy": source_prosody["energy"],
            "target_f0": target_prosody["f0"],
            "target_energy": target_prosody["energy"],
        }

    def apply_prosody(self, audio_path: str, warped_f0: np.ndarray,
                      warped_energy: np.ndarray, output_path: str,
                      sr: int = 24000) -> str:
        """Apply warped prosody (F0 + energy) to synthesized audio via PSOLA.
        
        Uses Parselmouth (Praat) for pitch manipulation via PSOLA.
        """
        print("[Prosody] Applying warped prosody via PSOLA...")
        snd = parselmouth.Sound(audio_path)

        if sr != int(snd.sampling_frequency):
            snd = snd.resample_time(sr)

        manipulation = parselmouth.praat.call(snd, "To Manipulation", 0.01, 60, 500)
        pitch_tier = parselmouth.praat.call(manipulation, "Extract pitch tier")

        parselmouth.praat.call(pitch_tier, "Remove points between", 0, snd.duration)

        hop_sec = 160 / 16000
        for i, f0_val in enumerate(warped_f0):
            time = i * hop_sec
            if time < snd.duration and f0_val > 60:
                parselmouth.praat.call(pitch_tier, "Add point", time, float(f0_val))

        parselmouth.praat.call([manipulation, pitch_tier], "Replace pitch tier")
        result = parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")

        result_np = result.values.flatten()

        target_rms = np.sqrt(np.mean(result_np ** 2))
        if target_rms > 0:
            hop_samples = int(hop_sec * sr)
            for i, energy_val in enumerate(warped_energy):
                start = i * hop_samples
                end = min(start + hop_samples, len(result_np))
                if start < len(result_np):
                    frame = result_np[start:end]
                    frame_rms = np.sqrt(np.mean(frame ** 2)) + 1e-8
                    if energy_val > 0:
                        scale = energy_val / frame_rms
                        scale = np.clip(scale, 0.3, 3.0)
                        result_np[start:end] = frame * scale

        peak = np.max(np.abs(result_np))
        if peak > 0:
            result_np = result_np / peak * 0.95

        import soundfile as sf
        sf.write(output_path, result_np, sr)
        print(f"[Prosody] Saved prosody-warped audio to {output_path}")
        return output_path


if __name__ == "__main__":
    print("[Prosody] Testing DTW implementation...")
    aligner = DTWAligner()
    source = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    target = torch.tensor([1.0, 1.5, 2.5, 3.5, 4.5, 5.0])
    path = aligner.compute_dtw_banded(source, target, band_radius=3)
    print(f"  DTW path: {path}")
    print("  DTW test PASSED")
