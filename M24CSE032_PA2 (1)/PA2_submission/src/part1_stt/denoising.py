"""Task 1.3: Denoising & Normalization using Spectral Subtraction.

Custom implementation of spectral subtraction for classroom audio denoising.
Includes pre-emphasis filtering and amplitude normalization.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import load_audio, save_audio, get_device


class SpectralSubtractionDenoiser:
    """Spectral Subtraction denoiser with pre-emphasis and normalization."""

    def __init__(self, n_fft: int = 400, hop_length: int = 160, win_length: int = 400,
                 noise_frames: int = 50, oversubtraction: float = 1.0,
                 spectral_floor: float = 0.002, pre_emphasis_coeff: float = 0.97):
        """
        Args:
            n_fft: FFT size (400 samples = 25ms at 16kHz)
            hop_length: Hop size (160 samples = 10ms at 16kHz)
            win_length: Window length
            noise_frames: Number of initial frames for noise estimation
            oversubtraction: Controls aggressiveness of noise removal
            spectral_floor: Minimum magnitude floor to avoid musical noise
            pre_emphasis_coeff: Pre-emphasis filter coefficient
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.noise_frames = noise_frames
        self.oversubtraction = oversubtraction
        self.spectral_floor = spectral_floor
        self.pre_emphasis_coeff = pre_emphasis_coeff
        self.device = get_device()

    def pre_emphasis(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]."""
        emphasized = torch.cat([
            waveform[..., :1],
            waveform[..., 1:] - self.pre_emphasis_coeff * waveform[..., :-1]
        ], dim=-1)
        return emphasized

    def de_emphasis(self, waveform: torch.Tensor) -> torch.Tensor:
        """Inverse pre-emphasis filter for reconstruction."""
        result = waveform.clone()
        for i in range(1, result.shape[-1]):
            result[..., i] = result[..., i] + self.pre_emphasis_coeff * result[..., i - 1]
        return result

    def estimate_noise_spectrum(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Estimate noise power spectrum from lowest-energy frames.
        
        Uses the bottom 10% energy frames (or initial silence frames) to
        robustly estimate the noise floor.
        """
        frame_energies = magnitude.pow(2).mean(dim=-2)
        n_noise = max(self.noise_frames, int(magnitude.shape[-1] * 0.1))
        _, low_energy_indices = torch.topk(frame_energies, n_noise, largest=False, dim=-1)
        
        noise_estimate = torch.zeros(magnitude.shape[:-1], device=magnitude.device)
        for idx in range(n_noise):
            frame_idx = low_energy_indices[..., idx]
            for b in range(magnitude.shape[0]) if magnitude.dim() > 2 else [0]:
                if magnitude.dim() > 2:
                    noise_estimate[b] += magnitude[b, :, frame_idx[b]].pow(2)
                else:
                    noise_estimate += magnitude[:, frame_idx].pow(2)
        noise_estimate /= n_noise
        return noise_estimate.sqrt()

    def spectral_subtraction(self, magnitude: torch.Tensor, phase: torch.Tensor,
                             noise_mag: torch.Tensor) -> torch.Tensor:
        """Apply spectral subtraction with spectral floor.
        
        Enhanced magnitude = max(|X| - alpha*|N|, beta*|X|)
        where alpha is oversubtraction factor and beta is spectral floor.
        """
        if noise_mag.dim() < magnitude.dim():
            noise_mag = noise_mag.unsqueeze(-1).expand_as(magnitude)

        subtracted = magnitude.pow(2) - self.oversubtraction * noise_mag.pow(2)
        floor = (self.spectral_floor * magnitude).pow(2)
        enhanced_power = torch.max(subtracted, floor)
        enhanced_mag = enhanced_power.sqrt()

        return enhanced_mag

    def process(self, waveform: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """Full denoising pipeline: pre-emphasis -> STFT -> spectral subtraction -> ISTFT -> de-emphasis."""
        waveform = waveform.to(self.device)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        emphasized = self.pre_emphasis(waveform)

        window = torch.hann_window(self.win_length, device=self.device)
        stft_out = torch.stft(
            emphasized.squeeze(0) if emphasized.shape[0] == 1 else emphasized,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True
        )

        magnitude = stft_out.abs()
        phase = stft_out.angle()

        noise_mag = self.estimate_noise_spectrum(magnitude)
        enhanced_mag = self.spectral_subtraction(magnitude, phase, noise_mag)

        enhanced_stft = enhanced_mag * torch.exp(1j * phase)

        enhanced_wav = torch.istft(
            enhanced_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window
        )

        if enhanced_wav.dim() == 1:
            enhanced_wav = enhanced_wav.unsqueeze(0)

        enhanced_wav = self.de_emphasis(enhanced_wav)

        peak = enhanced_wav.abs().max()
        if peak > 0:
            enhanced_wav = enhanced_wav / peak * 0.95

        return enhanced_wav.cpu()


def denoise_audio(input_path: str, output_path: str, sr: int = 16000) -> str:
    """Denoise an audio file and save the result."""
    print(f"[Denoising] Loading {input_path}...")
    waveform, sr = load_audio(input_path, sr=sr)

    denoiser = SpectralSubtractionDenoiser()
    print("[Denoising] Applying spectral subtraction...")
    denoised = denoiser.process(waveform, sr)

    save_audio(denoised, output_path, sr)
    print(f"[Denoising] Saved denoised audio to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="original_segment.wav")
    parser.add_argument("--output", default="denoised_segment.wav")
    args = parser.parse_args()
    denoise_audio(args.input, args.output)
