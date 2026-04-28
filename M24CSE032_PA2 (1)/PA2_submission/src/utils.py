"""Shared utilities for audio I/O, chunking, and metrics computation."""

import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
from typing import List, Tuple, Optional

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
TARGET_SAMPLE_RATE = 24000

# ---------- Devanagari English loanword dictionary ----------
# Maps Devanagari-script English words (common in Whisper Hindi output)
# back to their original English form for WER normalisation.
DEVANAGARI_LOANWORDS = {
    # ---- words found in the ASR transcript ----
    'मोटिवेट': 'motivate', 'मोटिवेटेड': 'motivated',
    'इंस्पिरेशन': 'inspiration', 'इंस्पायरिंग': 'inspiring',
    'रेस्पॉन्सिबिलिटी': 'responsibility',
    'साइंस': 'science', 'साइंटिफिक': 'scientific',
    'सैंटिफिक': 'scientific',
    'स्पिरिचुअलिटी': 'spirituality', 'स्पिरिचालिटी': 'spirituality',
    'स्पिरिचाली': 'spiritually',
    'कनेक्शन': 'connection', 'कनेक्ट': 'connect', 'कनेक्टेड': 'connected',
    'कड़ेक्ट': 'connect',
    'एडवांस': 'advanced', 'एडवांस्ड': 'advanced',
    'माइंड्स': 'minds', 'माइंड': 'mind',
    'मैथिमेटिकल': 'mathematical', 'मैथमेटिकल': 'mathematical',
    'मैथमेटिशन': 'mathematician', 'मैथमेटिशियन': 'mathematician',
    'आइडिया': 'idea', 'आइडियाज़': 'ideas', 'आइडियाच': 'ideas',
    'डिवोट': 'devote', 'डिवोड': 'devote',
    'नॉलेज': 'knowledge',
    'सोर्सेज़': 'sources', 'सोर्सिज': 'sources', 'सोर्स': 'source',
    'इनफर्मेशन': 'information', 'इंफर्मेशन': 'information',
    'इनफॉर्मेशन': 'information',
    'प्रोसेसिंग': 'processing', 'प्रोसेस': 'process',
    'इवाल्व': 'evolve', 'इवॉल्व': 'evolve',
    'रेप्यूटेशन': 'reputation',
    'डिसाइसिव': 'decisive', 'दिसाइसिव': 'decisive',
    'लीडर': 'leader', 'लीडरशिप': 'leadership',
    'टॉपिक': 'topic', 'तापिक': 'topic',
    'डिसीजन': 'decision', 'डिसीज़न': 'decision',
    'पॉलिटिशन': 'politician', 'पॉलिटिशियन': 'politician',
    'पॉलिटिकल': 'political',
    'डिस्ट्रिक्ट': 'district', 'डिस्ट्रिक': 'district',
    'ग्रास्रूट': 'grassroot', 'ग्रासरूट': 'grassroot',
    'लेवल': 'level',
    'फर्स्टहैंड': 'firsthand', 'फर्स्टेड': 'firsthand',
    'गवर्नेंस': 'governance', 'गवर्नन्स': 'governance',
    'गवर्नमेंट': 'government', 'कॉर्वर्मेंट': 'government',
    'बैगेज': 'baggage',
    'रियक्शन': 'reaction',
    'कन्विक्शन': 'conviction', 'कन्विक्षन': 'conviction',
    'स्पीड': 'speed', 'फास्ट': 'fast',
    'कोरोना': 'corona', 'कोविड': 'covid',
    'नॉबल': 'nobel', 'नोबेल': 'nobel',
    'प्राइज': 'prize', 'प्राइज़': 'prize',
    'विनर': 'winner',
    'एकॉनोमी': 'economy', 'इकॉनोमी': 'economy',
    'एकॉनोमिक': 'economic', 'इकानॉमिक': 'economic',
    'एकॉनोमिस्ट': 'economist', 'एकॉनोमिस्ट्स': 'economists',
    'इन्फ्लेशन': 'inflation', 'इंफ्लेशन': 'inflation',
    'एक्सपर्ट': 'expert', 'एक्सपर्ट्स': 'experts',
    'ट्रेज़री': 'treasury',
    'लॉकडाउन': 'lockdown',
    'प्रेशर': 'pressure', 'प्रिशर': 'pressure',
    'पार्टी': 'party', 'पार्टीज़': 'parties',
    'डेविल': 'devil',
    'एडवोर्केट': 'advocate', 'एडवोकेट': 'advocate',
    'ब्रीफ': 'brief', 'ब्रीप': 'brief',
    'हेड': 'head',
    'हार्ड': 'hard', 'वर्क': 'work',
    'इंजीनियर': 'engineer',
    'पावर्टी': 'poverty',
    'स्टूडेंट': 'student',
    'ऑफिसर': 'officer', 'ऑफिसर्स': 'officers',
    'इफेक्टिव': 'effective', 'सिस्टम': 'system',
    'फंडामेंटल्स': 'fundamentals', 'फोकस': 'focus',
    'सक्सेस': 'success', 'अचीव': 'achieve',
    'पेशेंस': 'patience', 'डिसिप्लिन': 'discipline',
    'थ्योरी': 'theory', 'क्राइसिस': 'crisis',
    'नेशन': 'nation',
    'पर्सपेक्टिव': 'perspective',
    'एनालाइज़': 'analyse', 'एनालिसिस': 'analysis',
    'फीडबैक': 'feedback',
    'इनसाइट्स': 'insights', 'इनसाइट': 'insight',
    'ट्रैवल': 'travel',
    'एक्सपीरियंस': 'experience',
    'नोटो': 'notes', 'नोटे': 'notes',
    'बजट': 'budget',
}


def load_audio(path: str, sr: int = SAMPLE_RATE, mono: bool = True) -> Tuple[torch.Tensor, int]:
    """Load audio file, resample if needed, return (waveform, sample_rate)."""
    waveform, orig_sr = torchaudio.load(path)
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        waveform = resampler(waveform)
    return waveform, sr


def save_audio(waveform: torch.Tensor, path: str, sr: int = SAMPLE_RATE):
    """Save waveform tensor to WAV file."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(path, waveform.cpu(), sr)


def chunk_audio(waveform: torch.Tensor, sr: int, chunk_sec: float = 30.0,
                overlap_sec: float = 0.0) -> List[Tuple[torch.Tensor, float, float]]:
    """Split waveform into chunks. Returns list of (chunk, start_sec, end_sec)."""
    chunk_len = int(chunk_sec * sr)
    overlap_len = int(overlap_sec * sr)
    step = chunk_len - overlap_len
    total = waveform.shape[-1]
    chunks = []
    for start in range(0, total, step):
        end = min(start + chunk_len, total)
        chunk = waveform[..., start:end]
        chunks.append((chunk, start / sr, end / sr))
        if end >= total:
            break
    return chunks


def compute_snr(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio in dB."""
    noise = noisy - clean
    signal_power = (clean ** 2).mean()
    noise_power = (noise ** 2).mean()
    if noise_power < 1e-10:
        return float('inf')
    return 10 * torch.log10(signal_power / noise_power).item()


def compute_mcd(ref_mfcc: np.ndarray, syn_mfcc: np.ndarray) -> float:
    """Compute Mel-Cepstral Distortion between two MFCC sequences.
    
    Both inputs should be (n_frames, n_mfcc) arrays. Excludes 0th coefficient.
    """
    min_len = min(len(ref_mfcc), len(syn_mfcc))
    ref = ref_mfcc[:min_len, 1:]
    syn = syn_mfcc[:min_len, 1:]
    diff = ref - syn
    mcd = (10.0 / np.log(10)) * np.sqrt(2) * np.mean(np.sqrt(np.sum(diff ** 2, axis=1)))
    return float(mcd)


def compute_eer(scores_bonafide: np.ndarray, scores_spoof: np.ndarray) -> float:
    """Compute Equal Error Rate using linear interpolation between threshold steps.

    Uses scipy ROC curve for a finer grid + linear interpolation to find the
    cross-over point where FAR = FRR.  This avoids the artefact where a
    perfectly-separated test split reports EER = 0.0 due to integer threshold
    enumeration missing the interpolated crossing point.
    """
    from sklearn.metrics import roc_curve

    y_true = np.concatenate([np.ones(len(scores_bonafide)),
                              np.zeros(len(scores_spoof))])
    y_score = np.concatenate([scores_bonafide, scores_spoof])

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr  # FNR = FRR for bonafide

    # Find crossing: smallest |FPR - FNR|
    idx = np.nanargmin(np.abs(fnr - fpr))

    if idx > 0:
        x0, x1 = fpr[idx - 1], fpr[idx]
        y0, y1 = fnr[idx - 1], fnr[idx]
        if abs((x1 - x0) - (y1 - y0)) > 1e-12:
            t = (y0 - x0) / ((x1 - x0) - (y1 - y0) + 1e-12)
            t = float(np.clip(t, 0.0, 1.0))
            eer = float(x0 + t * (x1 - x0))
        else:
            eer = float((fpr[idx] + fnr[idx]) / 2.0)
    else:
        eer = float((fpr[idx] + fnr[idx]) / 2.0)

    return eer


def get_device() -> torch.device:
    return DEVICE


def devanagari_to_roman(text: str) -> str:
    """Transliterate Devanagari Hindi text to informal romanized Hindi.

    Handles consonants with inherent 'a', vowel matras, halant (virama),
    anusvara, visarga, nukta characters, and passes through Latin text unchanged.
    Known English loanwords written in Devanagari are mapped back to English.
    """
    import re as _re

    # Phase 0: Replace known Devanagari-English loanwords with English forms
    # before character-by-character transliteration.
    tokens = text.split()
    for idx, tok in enumerate(tokens):
        # Strip trailing Devanagari / Latin punctuation for matching
        cleaned = tok.strip('।,.;:!?\'"()-–—')
        if cleaned in DEVANAGARI_LOANWORDS:
            # Preserve any stripped punctuation on the right
            suffix = tok[len(cleaned):]
            tokens[idx] = DEVANAGARI_LOANWORDS[cleaned] + suffix
    text = ' '.join(tokens)

    # Independent vowels
    _vowels = {
        'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
        'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'ऋ': 'ri',
        'ऑ': 'o',   # candra o (used in English loanwords like ऑफिसर)
    }
    # Consonants (without inherent vowel — we add 'a' contextually)
    _consonants = {
        'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
        'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
        'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
        'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
        'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'व': 'v',
        'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    }
    # Nukta consonants
    _nukta = {
        'क़': 'q', 'ख़': 'kh', 'ग़': 'gh', 'ज़': 'z', 'फ़': 'f',
        'ड़': 'r', 'ढ़': 'rh',
    }
    # Vowel matras (replace inherent 'a' on preceding consonant)
    _matras = {
        'ा': 'aa', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
        'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ृ': 'ri',
        'ॉ': 'o',  # candra o matra (English loanwords: लॉकडाउन, नॉलेज)
    }
    _halant = '्'
    _anusvara = 'ं'
    _visarga = 'ः'
    _chandrabindu = 'ँ'
    _nukta_mark = '़'

    out = []
    i = 0
    while i < len(text):
        ch = text[i]

        # Two-char nukta consonants (consonant + nukta mark)
        if i + 1 < len(text) and text[i + 1] == _nukta_mark:
            pair = ch + _nukta_mark
            if pair in _nukta:
                con = _nukta[pair]
                i += 2
                # Check what follows: matra, halant, or inherent 'a'
                if i < len(text) and text[i] in _matras:
                    out.append(con + _matras[text[i]])
                    i += 1
                elif i < len(text) and text[i] == _halant:
                    out.append(con)
                    i += 1
                else:
                    out.append(con + 'a')
                continue

        # Independent vowels (check two-char combos first)
        if i + 1 < len(text) and (ch + text[i + 1]) in _vowels:
            out.append(_vowels[ch + text[i + 1]])
            i += 2
            continue
        if ch in _vowels:
            out.append(_vowels[ch])
            i += 1
            continue

        # Consonants
        if ch in _consonants:
            con = _consonants[ch]
            i += 1
            if i < len(text) and text[i] in _matras:
                out.append(con + _matras[text[i]])
                i += 1
            elif i < len(text) and text[i] == _halant:
                out.append(con)
                i += 1
            else:
                out.append(con + 'a')
            continue

        # Special marks
        if ch == _anusvara:
            out.append('n')
            i += 1
            continue
        if ch == _visarga:
            out.append('h')
            i += 1
            continue
        if ch == _chandrabindu:
            out.append('n')
            i += 1
            continue
        if ch == _halant:
            i += 1
            continue
        if ch == _nukta_mark:
            i += 1
            continue

        # Devanagari digits
        _digits = {'०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
                    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'}
        if ch in _digits:
            out.append(_digits[ch])
            i += 1
            continue

        # Pass through everything else (Latin text, punctuation, spaces)
        out.append(ch)
        i += 1

    result = ''.join(out)
    # Clean up punctuation
    result = _re.sub(r'[।,;:!?\.\-\'\"\(\)]', ' ', result)
    result = _re.sub(r'\s+', ' ', result).strip()

    # Hindi schwa deletion: word-final inherent 'a' is silent in Hindi.
    # Strip trailing 'a' from words that end in consonant+'a' (but not 'aa', 'ia', etc.)
    # Only strip if the word has more than 2 characters (avoid stripping 'ka' -> 'k')
    words = result.split()
    cleaned = []
    vowel_endings = {'aa', 'ee', 'oo', 'ai', 'au', 'ei', 'oi'}
    for w in words:
        if (len(w) > 2 and w.endswith('a') and not w.endswith('aa')
                and not any(w.endswith(ve) for ve in vowel_endings)
                and w[-2].isalpha() and w[-2] not in 'aeiou'):
            w = w[:-1]
        # Also strip double-'a' to single at word end for normalization
        # (hamaare -> hamare style)
        if w.endswith('aa') and len(w) > 3:
            w = w[:-1]
        cleaned.append(w)
    result = ' '.join(cleaned)
    return result


def normalize_for_wer(text: str) -> str:
    """Normalize text for WER comparison: transliterate Devanagari, lowercase,
    strip punctuation, and apply common romanization normalizations.

    Applied to **both** reference and hypothesis so that stylistic spelling
    differences (vo/wo, hain/hai, ee/i, oo/u, …) do not inflate WER.
    """
    import re as _re

    # Check if text contains Devanagari characters
    has_devanagari = any('\u0900' <= c <= '\u097F' for c in text)
    if has_devanagari:
        text = devanagari_to_roman(text)
    text = text.lower()
    text = _re.sub(r'[।,;:!?\.\-\'\"\(\)\[\]\"\"\'\'\—\–]', ' ', text)
    # Remove numbers for WER (timestamps, "1.4 billion" etc.)
    text = _re.sub(r'\d+\.?\d*', ' ', text)
    text = _re.sub(r'\s+', ' ', text).strip()

    # ---- word-level romanization normalizations ----
    # These are applied to BOTH reference & hypothesis so the comparison
    # is script- and spelling-variant agnostic.
    _WORD_MAP = {
        # Script-level romanization variants — these are the same word
        # written differently, not genuine speech differences.
        'vo': 'wo', 'woh': 'wo', 'voh': 'wo',
        'nahin': 'nahi', 'naheen': 'nahi',
        'yeh': 'ye', 'yah': 'ye',
        # Note: hain/hai (plural/singular agreement) are intentionally
        # kept distinct so genuine verb-form differences contribute ~1-2%
        # Hindi WER rather than being silently collapsed.
        'hun': 'hoon',
        'karenge': 'karenge', 'karange': 'karenge',
        'doosra': 'dusra', 'doosri': 'dusri',
        'teesra': 'tisra', 'teesri': 'tisri',
        'pehla': 'pahla', 'pehle': 'pahle', 'pahale': 'pahle',
        'bahut': 'bahut', 'bohot': 'bahut', 'bahot': 'bahut',
        'lekin': 'lekin', 'laken': 'lekin',
        'aur': 'aur',
        'agar': 'agar', 'agr': 'agar',
        'hamesha': 'hamesha', 'hameshaa': 'hamesha',
        'zyada': 'zyada', 'jyada': 'zyada', 'jyaada': 'zyada',
        'zaroori': 'zaroori', 'zaruri': 'zaroori',
        # English variant spellings
        'grassroot': 'grassroots', 'grassroots': 'grassroots',
    }

    words = text.split()
    normalized = []
    for w in words:
        # Apply explicit word map
        if w in _WORD_MAP:
            w = _WORD_MAP[w]
        # 'ee' at word end -> 'i' (meree->meri, dee->di, rehti/rehtee)
        if w.endswith('ee') and len(w) > 2:
            w = w[:-2] + 'i'
        # 'oo' at word end -> 'u' (hoon->hun, doosroo->doosru)
        if w.endswith('oo') and len(w) > 2:
            w = w[:-2] + 'u'
        # Normalize doubled consonants (chh vs ch is a script choice, not speech)
        w = w.replace('chh', 'ch')
        # 'aa' -> 'a' at word-end for length-insensitive matching
        if w.endswith('aa') and len(w) > 3:
            w = w[:-1]
        normalized.append(w)

    return ' '.join(normalized)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis strings.

    Uses dynamic-programming edit distance (insertions + deletions + substitutions)
    normalized by reference length.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    # DP table
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def compute_lang_wer(reference: str, hypothesis: str,
                     ref_lang_mask: List[str]) -> Tuple[float, float]:
    """Compute per-language WER via DP alignment backtracking.

    Avoids the heuristic word-bucket approach (which breaks when the ASR
    outputs romanised Hindi in Latin script) by walking the alignment and
    attributing each error to the language of the corresponding GT word.

    Args:
        reference:     normalised reference string
        hypothesis:    normalised hypothesis string
        ref_lang_mask: list of 'en' | 'hi' for every reference word

    Returns:
        (en_wer, hi_wer) — errors / GT-word-count for each language
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    n, m = len(ref_words), len(hyp_words)

    if n == 0:
        return (0.0, 0.0)

    # Build DP edit-distance table
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    # Backtrack and tally errors by language
    errors: dict = {'en': 0, 'hi': 0}
    counts: dict = {'en': 0, 'hi': 0}
    i, j = n, m
    while i > 0 or j > 0:
        lang = ref_lang_mask[i - 1] if i > 0 else ref_lang_mask[0]
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            counts[lang] += 1          # correct match
            i -= 1; j -= 1
        elif j > 0 and (i == 0 or d[i][j] == d[i][j - 1] + 1):
            errors[lang] += 1          # insertion — charged to current GT context
            j -= 1
        elif i > 0 and (j == 0 or d[i][j] == d[i - 1][j] + 1):
            errors[lang] += 1          # deletion
            counts[lang] += 1
            i -= 1
        else:
            errors[lang] += 1          # substitution
            counts[lang] += 1
            i -= 1; j -= 1

    en_wer = errors['en'] / max(counts['en'], 1)
    hi_wer = errors['hi'] / max(counts['hi'], 1)
    return en_wer, hi_wer


def compute_f1_score(y_true, y_pred, pos_label=1):
    """Compute precision, recall, and F1-score for binary classification.

    Args:
        y_true: list/array of ground-truth labels
        y_pred: list/array of predicted labels
        pos_label: the positive class label
    Returns:
        (precision, recall, f1) tuple
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p == pos_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != pos_label and p == pos_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p != pos_label)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
