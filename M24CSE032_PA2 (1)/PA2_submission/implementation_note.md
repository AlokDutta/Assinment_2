# Implementation Note: Non-Obvious Design Choices

**Roll No:** M24CSE032 | **Course:** Speech Understanding — Programming Assignment 2

---

## Part I: Class-Balanced Pseudo-Label Training for Frame-Level LID

**Design Choice:** Distilling VoxLingua107's coarse language predictions into a fine-grained Multi-Head Attention classifier, trained with **class-balanced resampling + inverse-frequency-weighted cross-entropy**, rather than using VoxLingua107 directly or applying vanilla CE.

**Rationale:** VoxLingua107 is trained for utterance-level classification across 107 languages; its predictions on 400 ms windows are noisy and lack temporal coherence. Distillation into a lightweight 2-class attention classifier gives (i) temporal smoothing via learned attention weights over 5-frame contexts and (ii) a differentiable classifier that enables gradient-based adversarial attacks in Part IV — VoxLingua107's `encode_batch` blocks gradients, making FGSM impossible without this architectural separation. However, the source audio is ~90 % Hindi / ~10 % English, so vanilla cross-entropy collapses the classifier to a majority-class predictor (English F1 ≈ 0.39, macro F1 ≈ 0.69 — below the 0.85 bar). We fix this with two complementary mechanisms: each epoch draws a **50/50 balanced index set** (oversampling English with replacement to match the Hindi count), and `F.cross_entropy` uses **weights = N/(2·n_c)** so that the minority class also contributes proportionally per sample. The two mechanisms are additive — resampling exposes the model to more English batches while weights keep the per-batch gradient direction unbiased.

---

## Part II: Devanagari Language Bridge for Meitei Translation

**Design Choice:** Transliterating Meitei text into Devanagari script for TTS synthesis instead of attempting direct Meitei Mayek rendering.

**Rationale:** No current TTS model supports Meitei natively. Rather than fine-tuning (which requires paired data we don't have), we exploit the phonological overlap between Meitei and Hindi: both have similar vowel systems and share retroflexes (/ʈ/, /ɖ/) absent in European languages. By rendering Meitei words in Devanagari, XTTS v2's Hindi mode produces phonetically closer output than any English-mode approximation would. This "language bridge" sacrifices tonal accuracy (Meitei is tonal; Hindi is not) but preserves segmental intelligibility — a pragmatic tradeoff for zero-resource synthesis.

---

## Part III: Sakoe-Chiba Band DTW with Downsampling for Prosody Warping

**Design Choice:** Downsampling 60,000-frame prosody contours to 5,000 frames before DTW, using a Sakoe-Chiba band constraint of width R = max(50, N/5), then mapping the alignment path back via index scaling.

**Rationale:** Naive DTW on two 60K-frame sequences requires a 60K×60K cost matrix (27 GB), causing OOM on any GPU. Even on CPU, the O(N²) double-loop would take hours. Our three-layer optimization — (1) averaging-based downsampling, (2) banded computation reducing O(N²) to O(N·R), and (3) CPU execution to avoid GPU memory pressure during TTS model residency — reduces compute to ~8 minutes while preserving the global alignment structure. The key insight is that prosodic features vary slowly (F0 changes at ~10 Hz rate), so 12:1 downsampling loses negligible alignment accuracy.

---

## Part IV: Early-Stopped I-FGSM Through a Differentiable ECAPA-TDNN

**Design Choice:** Bypassing SpeechBrain's `encode_batch` (which wraps inference in `torch.no_grad()`) by directly calling `ecapa.mods.compute_features` → **`ecapa.mods.mean_var_norm`** → `ecapa.mods.embedding_model` to maintain gradient flow, plus early-stopping once ≥50% of frames flip, with per-step size $\alpha = \epsilon / n_\text{iter}$.

**Rationale:** A proxy-based transfer attack fails — gradient directions from a simple CNN proxy do not transfer to ECAPA-TDNN's deep architecture. Using real model internals gets gradient flow, but two non-obvious bugs were encountered: (1) **missing `mean_var_norm`** — SpeechBrain's encode_batch pipeline is Fbank → mean_var_norm → ECAPA-TDNN; skipping mean_var_norm means the embedding model receives unnormalised features, driving the waveform gradient to ≈10⁻¹² (effectively zero). With near-zero gradient, sign(grad) gives random perturbation directions, making the attack degrade to random noise injection and requiring ε > 0.02 for a flip (SNR < 40 dB). Adding mean_var_norm raised gradient magnitude to 1.37×10³ and the minimum flip epsilon dropped to **0.003144 (SNR = 69.5 dB)**. (2) **Early stopping preserves SNR** — without it, I-FGSM keeps accumulating perturbation past the flip point, wasting 20-30 dB of SNR budget on unnecessary iterations.
