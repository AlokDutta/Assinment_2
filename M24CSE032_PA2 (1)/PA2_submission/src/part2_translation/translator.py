"""Task 2.2: Semantic Translation to Meitei (Manipuri).

Translates English/Hindi transcripts to Meitei using a 500-word parallel corpus,
with transliteration fallback for technical terms.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional


DEVANAGARI_TO_MEITEI_MAYEK = {
    "अ": "ꯑ", "आ": "ꯑꯥ", "इ": "ꯏ", "ई": "ꯏꯏ", "उ": "ꯎ", "ऊ": "ꯎꯎ",
    "ए": "ꯑꯦ", "ऐ": "ꯑꯩ", "ओ": "ꯑꯣ", "औ": "ꯑꯧ",
    "क": "ꯀ", "ख": "ꯈ", "ग": "ꯒ", "घ": "ꯘ",
    "च": "ꯆ", "छ": "ꯆꯍ", "ज": "ꯖ", "झ": "ꯖꯍ",
    "ट": "ꯇ", "ठ": "ꯊ", "ड": "ꯗ", "ढ": "ꯙ", "ण": "ꯅ",
    "त": "ꯇ", "थ": "ꯊ", "द": "ꯗ", "ध": "ꯙ", "न": "ꯅ",
    "प": "ꯄ", "फ": "ꯐ", "ब": "ꯕ", "भ": "ꯚ", "म": "ꯃ",
    "य": "ꯌ", "र": "ꯔ", "ल": "ꯂ", "व": "ꯋ",
    "श": "ꯁ", "ष": "ꯁ", "स": "ꯁ", "ह": "ꯍ",
    "ं": "ꯪ", "ः": "ꯍ", "ँ": "ꯪ",
    "ा": "ꯥ", "ि": "ꯤ", "ी": "ꯤꯤ", "ु": "ꯨ", "ू": "ꯨꯨ",
    "े": "ꯦ", "ै": "ꯩ", "ो": "ꯣ", "ौ": "ꯧ",
    "्": "", "़": "",
}

LATIN_TO_DEVANAGARI = {
    "a": "अ", "aa": "आ", "i": "इ", "ee": "ई", "u": "उ", "oo": "ऊ",
    "e": "ए", "ai": "ऐ", "o": "ओ", "au": "औ",
    "ka": "क", "kha": "ख", "ga": "ग", "gha": "घ",
    "cha": "च", "chha": "छ", "ja": "ज", "jha": "झ",
    "ta": "ट", "tha": "ठ", "da": "ड", "dha": "ढ", "na": "ण",
    "pa": "प", "pha": "फ", "ba": "ब", "bha": "भ", "ma": "म",
    "ya": "य", "ra": "र", "la": "ल", "va": "व", "wa": "व",
    "sha": "श", "sa": "स", "ha": "ह",
    "k": "क्", "g": "ग्", "t": "त्", "d": "द्", "n": "न",
    "p": "प्", "b": "ब्", "m": "म", "r": "र", "l": "ल",
    "s": "स", "h": "ह",
}


class MeiteiTranslator:
    """Translates English/Hindi text to Meitei using parallel corpus + transliteration."""

    def __init__(self, corpus_path: Optional[str] = None):
        if corpus_path is None:
            corpus_path = str(Path(__file__).parent / "parallel_corpus.json")

        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)

        self.entries = corpus_data["entries"]
        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """Build fast lookup dictionaries from the parallel corpus."""
        self.en_to_mni = {}
        self.en_to_mni_latin = {}
        self.hi_to_mni = {}
        self.hi_to_mni_latin = {}

        for entry in self.entries:
            en = entry["en"].lower()
            mni_latin = entry.get("mni_latin", "")
            self.en_to_mni[en] = entry.get("mni", "")
            self.en_to_mni_latin[en] = mni_latin

            hi = entry.get("hi", "")
            if hi:
                self.hi_to_mni[hi] = entry.get("mni", "")
                self.hi_to_mni_latin[hi] = mni_latin

        print(f"[Translator] Loaded {len(self.en_to_mni)} EN->MNI entries, "
              f"{len(self.hi_to_mni)} HI->MNI entries")

    def translate_word(self, word: str, src_lang: str = "en") -> Dict[str, str]:
        """Translate a single word to Meitei.
        
        Returns dict with keys: mni (Meitei Mayek), mni_latin (Latin transliteration),
        devanagari (Devanagari rendering for TTS).
        
        For corpus matches the devanagari field contains the Meitei word
        rendered in Devanagari so XTTS speaks Meitei phonemes, not Hindi.
        """
        word_lower = word.lower().strip()

        if src_lang == "en" and word_lower in self.en_to_mni:
            mni_lat = self.en_to_mni_latin.get(word_lower, word_lower)
            return {
                "mni": self.en_to_mni[word_lower],
                "mni_latin": mni_lat,
                "devanagari": self._latin_to_devanagari(mni_lat),
            }

        if src_lang == "hi" and word in self.hi_to_mni:
            mni_lat = self.hi_to_mni_latin.get(word, word_lower)
            return {
                "mni": self.hi_to_mni[word],
                "mni_latin": mni_lat,
                "devanagari": self._latin_to_devanagari(mni_lat),
            }

        if src_lang == "hi":
            mni_mayek = self._devanagari_to_meitei(word)
            return {
                "mni": mni_mayek,
                "mni_latin": word_lower,
                "devanagari": word,
            }

        transliterated = self._transliterate_to_meitei(word_lower)
        return {
            "mni": transliterated,
            "mni_latin": word_lower,
            "devanagari": self._latin_to_devanagari(word_lower),
        }

    def translate_sentence(self, sentence: str, src_lang: str = "en") -> Dict[str, str]:
        """Translate a full sentence to Meitei."""
        words = sentence.split()
        mni_parts = []
        latin_parts = []
        deva_parts = []

        i = 0
        while i < len(words):
            matched = False
            for length in range(min(3, len(words) - i), 0, -1):
                phrase = " ".join(words[i:i + length]).lower()
                if phrase in self.en_to_mni:
                    result = self.translate_word(phrase, src_lang)
                    mni_parts.append(result["mni"])
                    latin_parts.append(result["mni_latin"])
                    deva_parts.append(result["devanagari"])
                    i += length
                    matched = True
                    break

            if not matched:
                result = self.translate_word(words[i], src_lang)
                mni_parts.append(result["mni"])
                latin_parts.append(result["mni_latin"])
                deva_parts.append(result["devanagari"])
                i += 1

        return {
            "mni": " ".join(mni_parts),
            "mni_latin": " ".join(latin_parts),
            "devanagari": " ".join(deva_parts),
        }

    def translate_transcript(self, transcripts: List[Dict]) -> List[Dict]:
        """Translate full transcript segments to Meitei."""
        translated = []
        for seg in transcripts:
            text = seg.get("text", "")
            lang = seg.get("language", "en")
            result = self.translate_sentence(text, src_lang=lang)
            translated.append({
                **seg,
                "mni_text": result["mni"],
                "mni_latin": result["mni_latin"],
                "devanagari": result["devanagari"],
            })
        return translated

    def _devanagari_to_meitei(self, word: str) -> str:
        """Convert a Devanagari word to Meitei Mayek character by character."""
        meitei = ""
        for char in word:
            meitei += DEVANAGARI_TO_MEITEI_MAYEK.get(char, char)
        return meitei

    def _transliterate_to_meitei(self, word: str) -> str:
        """Transliterate a Latin-script word to Meitei Mayek script."""
        devanagari = self._latin_to_devanagari(word)
        return self._devanagari_to_meitei(devanagari)

    def _latin_to_devanagari(self, word: str) -> str:
        """Convert Latin transliteration to Devanagari script for TTS input."""
        result = []
        i = 0
        word = word.lower()
        while i < len(word):
            matched = False
            for length in [3, 2, 1]:
                if i + length <= len(word):
                    chunk = word[i:i + length]
                    if chunk in LATIN_TO_DEVANAGARI:
                        result.append(LATIN_TO_DEVANAGARI[chunk])
                        i += length
                        matched = True
                        break
            if not matched:
                result.append(word[i])
                i += 1
        return "".join(result)

    def get_devanagari_text(self, transcripts: List[Dict]) -> str:
        """Get concatenated Devanagari text for TTS synthesis."""
        translated = self.translate_transcript(transcripts)
        parts = []
        for seg in translated:
            deva = seg.get("devanagari", "")
            if deva.strip():
                parts.append(deva)
        return " । ".join(parts)


def translate_to_meitei(transcripts: List[Dict]) -> List[Dict]:
    """Translate transcripts to Meitei."""
    translator = MeiteiTranslator()
    translated = translator.translate_transcript(transcripts)
    print(f"[Translation] Translated {len(translated)} segments to Meitei")
    for seg in translated[:3]:
        print(f"  EN/HI: {seg.get('text', '')[:60]}...")
        print(f"  MNI:   {seg.get('mni_text', '')[:60]}...")
    return translated


if __name__ == "__main__":
    test = [
        {"text": "the speech signal has high frequency", "language": "en"},
        {"text": "yeh spectral analysis hai", "language": "hi"},
    ]
    result = translate_to_meitei(test)
    for r in result:
        print(f"\nOriginal: {r['text']}")
        print(f"Meitei: {r['mni_text']}")
        print(f"Latin: {r['mni_latin']}")
        print(f"Devanagari: {r['devanagari']}")
