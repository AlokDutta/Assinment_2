"""Task 2.1: IPA Unified Representation for Hinglish.

Converts code-switched English-Hindi transcripts into unified IPA strings.
Uses epitran for base G2P with a custom Hinglish mapping layer for romanized
Hindi words and code-switch boundaries.
"""

import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path


ROMANIZED_HINDI_TO_IPA = {
    "a": "ə", "aa": "aː", "i": "ɪ", "ii": "iː", "ee": "iː",
    "u": "ʊ", "uu": "uː", "oo": "uː", "e": "eː", "ai": "ɛː",
    "o": "oː", "au": "ɔː", "ou": "ɔː",
    "ka": "kə", "kha": "kʰə", "ga": "ɡə", "gha": "ɡʱə",
    "cha": "t͡ʃə", "chha": "t͡ʃʰə", "ja": "d͡ʒə", "jha": "d͡ʒʱə",
    "ta": "ʈə", "tha": "ʈʰə", "da": "ɖə", "dha": "ɖʱə",
    "na": "nə", "pa": "pə", "pha": "pʰə", "fa": "fə",
    "ba": "bə", "bha": "bʱə", "ma": "mə",
    "ya": "jə", "ra": "ɾə", "la": "lə", "va": "ʋə", "wa": "ʋə",
    "sha": "ʃə", "sa": "sə", "ha": "ɦə",
    "k": "k", "kh": "kʰ", "g": "ɡ", "gh": "ɡʱ",
    "ch": "t͡ʃ", "j": "d͡ʒ", "jh": "d͡ʒʱ",
    "t": "t̪", "th": "t̪ʰ", "d": "d̪", "dh": "d̪ʱ",
    "n": "n", "p": "p", "ph": "pʰ", "f": "f",
    "b": "b", "bh": "bʱ", "m": "m",
    "y": "j", "r": "ɾ", "l": "l", "v": "ʋ", "w": "ʋ",
    "sh": "ʃ", "s": "s", "h": "ɦ",
    "ng": "ŋ", "ny": "ɲ",
}

HINGLISH_LEXICON = {
    "accha": "ət͡ʃːʰaː", "acha": "ət͡ʃːʰaː",
    "kaise": "kɛːseː", "kaisa": "kɛːsaː",
    "haan": "ɦaːn", "nahi": "nəɦiː", "nahin": "nəɦĩː",
    "kya": "kjaː", "kyun": "kjũː", "kyu": "kjuː",
    "hai": "ɦɛː", "hain": "ɦɛ̃ː", "tha": "t̪ʰaː",
    "toh": "t̪oː", "to": "t̪oː", "bhi": "bʱiː",
    "aur": "ɔːɾ", "ya": "jaː", "ki": "kiː",
    "ko": "koː", "se": "seː", "ka": "kaː",
    "ke": "keː", "mein": "mẽː", "me": "meː",
    "par": "pəɾ", "pe": "peː", "woh": "ʋoː",
    "yeh": "jeː", "ye": "jeː", "ek": "eːk",
    "do": "d̪oː", "teen": "t̪iːn", "char": "t͡ʃaːɾ",
    "matlab": "mət̪ləb", "samajh": "səməd͡ʒʱ",
    "samjho": "səmd͡ʒʱoː", "dekho": "d̪eːkʰoː",
    "suno": "sʊnoː", "bolo": "boːloː",
    "wala": "ʋaːlaː", "wali": "ʋaːliː", "wale": "ʋaːleː",
    "lekin": "leːkɪn", "agar": "əɡəɾ", "jab": "d͡ʒəb",
    "tab": "t̪əb", "abhi": "əbʱiː", "bahut": "bəɦʊt̪",
    "thoda": "t̪ʰoːɖaː", "zyada": "zjaːd̪aː",
    "pehle": "pəɦleː", "baad": "baːd̪",
    "signal": "sɪɡnəl", "frequency": "fɾiːkʋənsiː",
    "spectrum": "spɛkt̪ɾəm", "filter": "fɪlt̪əɾ",
    "matlab": "mət̪ləb", "processor": "pɾoːsɛsəɾ",
    "isko": "ɪskoː", "usko": "ʊskoː", "inko": "ɪnkoː",
    "karo": "kəɾoː", "karna": "kəɾnaː", "karke": "kəɾkeː",
    "hota": "ɦoːt̪aː", "hoti": "ɦoːt̪iː", "hote": "ɦoːt̪eː",
    "jaata": "d͡ʒaːt̪aː", "aata": "aːt̪aː",
    "padta": "pəɖt̪aː", "milta": "mɪlt̪aː",
    "sakte": "sɛkt̪eː", "sakta": "sɛkt̪aː",
    "chahiye": "t͡ʃaːɦɪjeː", "zaruri": "zəɾuːɾiː",
}

TECHNICAL_TERMS_IPA = {
    "mfcc": "ɛm.ɛf.siː.siː",
    "hmm": "eːt͡ʃ.ɛm.ɛm",
    "gmm": "d͡ʒiː.ɛm.ɛm",
    "dft": "d̪iː.ɛf.t̪iː",
    "fft": "ɛf.ɛf.t̪iː",
    "stft": "ɛs.t̪iː.ɛf.t̪iː",
    "lpc": "ɛl.piː.siː",
    "rnn": "aːɾ.ɛn.ɛn",
    "lstm": "ɛl.ɛs.t̪iː.ɛm",
    "gru": "d͡ʒiː.aːɾ.juː",
    "cnn": "siː.ɛn.ɛn",
    "ctc": "siː.t̪iː.siː",
    "asr": "eː.ɛs.aːɾ",
    "tts": "t̪iː.t̪iː.ɛs",
    "ipa": "aɪ.piː.eː",
    "wer": "ʋeː.iː.aːɾ",
    "snr": "ɛs.ɛn.aːɾ",
    "eer": "iː.iː.aːɾ",
    "dtw": "d̪iː.t̪iː.ʋeː",
    "g2p": "d͡ʒiː.t̪uː.piː",
    "lfcc": "ɛl.ɛf.siː.siː",
    "cqcc": "siː.kjuː.siː.siː",
    "fgsm": "ɛf.d͡ʒiː.ɛs.ɛm",
}


class HinglishIPAConverter:
    """Converts Hinglish (code-switched English-Hindi) text to unified IPA."""

    def __init__(self):
        self._init_epitran()
        self.hinglish_lexicon = HINGLISH_LEXICON
        self.romanized_map = ROMANIZED_HINDI_TO_IPA
        self.technical_terms = TECHNICAL_TERMS_IPA

    def _init_epitran(self):
        """Initialize epitran G2P for English and Hindi."""
        try:
            import epitran
            self.epi_en = epitran.Epitran('eng-Latn')
            self.epi_hi = epitran.Epitran('hin-Deva')
            self._epitran_available = True
        except Exception as e:
            print(f"[IPA] Warning: epitran init failed ({e}), using fallback")
            self._epitran_available = False

    def convert_word(self, word: str, lang: str = "en") -> str:
        """Convert a single word to IPA.
        
        Priority: technical terms > Hinglish lexicon > epitran G2P > romanized rules
        """
        word_lower = word.lower().strip()

        if word_lower in self.technical_terms:
            return self.technical_terms[word_lower]

        if word_lower in self.hinglish_lexicon:
            return self.hinglish_lexicon[word_lower]

        if "-" in word_lower:
            parts = word_lower.split("-")
            return ".".join(self.convert_word(p, lang) for p in parts if p)

        if self._epitran_available:
            try:
                if lang == "hi":
                    return self.epi_hi.transliterate(word)
                else:
                    return self.epi_en.transliterate(word)
            except Exception:
                pass

        return self._romanized_to_ipa(word_lower)

    def _romanized_to_ipa(self, word: str) -> str:
        """Convert romanized Hindi/English word to IPA using rule-based mapping."""
        result = []
        i = 0
        while i < len(word):
            matched = False
            for length in [3, 2, 1]:
                if i + length <= len(word):
                    chunk = word[i:i + length]
                    if chunk in self.romanized_map:
                        result.append(self.romanized_map[chunk])
                        i += length
                        matched = True
                        break
            if not matched:
                result.append(word[i])
                i += 1
        return "".join(result)

    def convert_segment(self, text: str, lang: str = "en") -> str:
        """Convert a text segment to IPA."""
        words = re.findall(r"[\w'-]+|[^\w\s]", text)
        ipa_parts = []
        for word in words:
            if re.match(r'^[^\w]$', word):
                ipa_parts.append(word)
            else:
                ipa = self.convert_word(word, lang)
                ipa_parts.append(ipa)
        return " ".join(ipa_parts)

    def convert_transcript(self, transcripts: List[Dict]) -> str:
        """Convert full transcript with language tags to unified IPA.
        
        Args:
            transcripts: list of {"text": str, "language": str, ...}
        Returns:
            Unified IPA string with language markers
        """
        ipa_segments = []
        for seg in transcripts:
            text = seg.get("text", "")
            lang = seg.get("language", "en")

            words = text.split()
            ipa_words = []
            for word in words:
                clean = re.sub(r'[^\w-]', '', word)
                if not clean:
                    continue

                if self._is_romanized_hindi(clean) and lang == "en":
                    ipa = self.convert_word(clean, "hi")
                else:
                    ipa = self.convert_word(clean, lang)
                ipa_words.append(ipa)

            ipa_text = " ".join(ipa_words)
            ipa_segments.append(f"[{lang}] {ipa_text}")

        return "\n".join(ipa_segments)

    def _is_romanized_hindi(self, word: str) -> bool:
        """Detect if a word is likely romanized Hindi."""
        word_lower = word.lower()
        if word_lower in self.hinglish_lexicon:
            return True

        hindi_suffixes = ["na", "ni", "ne", "ka", "ki", "ke", "ta", "ti", "te",
                          "wala", "wali", "wale", "iya", "iye"]
        for suffix in hindi_suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 1:
                return True

        return False


def convert_to_ipa(transcripts: List[Dict]) -> str:
    """Convert transcripts to unified IPA representation."""
    converter = HinglishIPAConverter()
    ipa_text = converter.convert_transcript(transcripts)
    print(f"[IPA] Converted {len(transcripts)} segments to IPA")
    print(f"[IPA] Preview: {ipa_text[:300]}...")
    return ipa_text


if __name__ == "__main__":
    test_transcripts = [
        {"text": "So the MFCC features are extracted from the spectrogram", "language": "en"},
        {"text": "toh yeh frequency domain mein hota hai", "language": "hi"},
        {"text": "the stochastic process and hidden markov model", "language": "en"},
    ]
    ipa = convert_to_ipa(test_transcripts)
    print("\n" + ipa)
