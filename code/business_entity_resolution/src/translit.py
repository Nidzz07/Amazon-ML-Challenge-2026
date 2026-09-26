"""Script detection and romanisation (Track C). Pure: no I/O, no module-level state.

Every Indic script in the data (9 of them) is romanised through one path. Each
character is moved onto the Devanagari block by its Unicode offset. The Brahmic
blocks share the ISCII layout, so this is exact for everything except a handful of
script-specific letters, which are remapped first. The result is transliterated
Devanagari -> IAST with indic-transliteration, then IAST is folded to plain ASCII.
The offset trick also fixes Tamil, which the library's own Tamil scheme reads as
Grantha-annotated Sanskrit (சதர்ன் "southern" comes out as "jhadharṉ").

Schwa deletion: Devanagari, Bengali, Gurmukhi, Gujarati and Oriya leave the
word-final inherent vowel unwritten but unpronounced (लिमिटेड = limited, not
limiteda), so a virama is added after a word-final bare consonant. Tamil, Telugu,
Kannada and Malayalam write the virama explicitly and are left alone. Medial schwa
is not handled; fuzzy matching absorbs it.

Romanisation is memoised per Indic run (a maximal run of Indic characters, which is
one word) in a dict owned by a Romaniser instance. Business names reuse a small
vocabulary (प्राइवेट, लिमिटेड, ...), so almost every lookup is a cache hit.
"""
import regex
from indic_transliteration import sanscript

SCRIPT_CODES = {
    "none": 0, "latin": 1, "devanagari": 2, "bengali": 3, "gurmukhi": 4, "gujarati": 5,
    "oriya": 6, "tamil": 7, "telugu": 8, "kannada": 9, "malayalam": 10, "other": 15,
}
# Unicode block base -> script; every block is 0x80 wide, Devanagari first.
BLOCK_SCRIPTS = ("devanagari", "bengali", "gurmukhi", "gujarati", "oriya", "tamil", "telugu", "kannada", "malayalam")
INDIC_FIRST, INDIC_LAST = 0x0900, 0x0DFF
# Polars/Rust-regex patterns, shared with normalise.py.
INDIC_PATTERN = r"[ऀ-෿]"
OTHER_SCRIPT_PATTERN = r"[\p{L}--\p{Latin}]"
LATIN_PATTERN = r"\p{Latin}"

SCHWA_DELETING = {"devanagari", "bengali", "gurmukhi", "gujarati", "oriya"}
_INDIC_RUN = regex.compile(r"[ऀ-෿][ऀ-෿‌‍]*")
_OTHER = regex.compile(OTHER_SCRIPT_PATTERN, regex.V1)
_LATIN = regex.compile(LATIN_PATTERN)

# Letters with no same-offset Devanagari equivalent, mapped before the offset shift.
_PRE = {
    "ऑ": "ओ", "ॉ": "ो",  # candra O (ऑ, ॉ, "auto") -> O
    "ऍ": "ए", "ॅ": "े",  # candra E -> E
    "ৎ": "त्",  # Bengali khanda ta -> t
    "ੰ": "ਂ", "ੱ": "",  # Gurmukhi tippi -> bindi; addak (gemination) dropped
    "ൺ": "ण्", "ൻ": "न्", "ർ": "र्",  # Malayalam chillus
    "ൽ": "ल्", "ൾ": "ळ्", "ൿ": "क्",
    "‌": "", "‍": "",
}
_CONSONANT = set(range(0x0915, 0x093A)) | set(range(0x0958, 0x0960))
_NUKTA, _VIRAMA = 0x093C, "्"

# IAST -> ASCII. Order matters: multi-character keys first.
_IAST_SUBS = [
    (regex.compile(r"ṟṟ"), "tt"),  # Malayalam റ്റ
    (regex.compile(r"ṃ(?=[pbm])"), "m"),
    (regex.compile(r"c(h?)"), lambda m: "chh" if m.group(1) else "ch"),  # IAST c = च, ch = छ
]
_IAST_CHARS = str.maketrans({
    "ā": "a", "ī": "i", "ū": "u", "ē": "e", "ō": "o", "è": "e", "ò": "o",
    "ṭ": "t", "ḍ": "d", "ṇ": "n", "ñ": "n", "ṅ": "n", "ṉ": "n", "ṃ": "n", "~": "n",
    "ś": "sh", "ṣ": "sh", "ḥ": "h", "ṛ": "ri", "ṝ": "ri", "ḷ": "l", "ḻ": "l", "ṟ": "r",
    "ẏ": "y", "|": " ", "̤": "", "̐": "n",  # r̤ / l̤ diaeresis-below dropped; m̐ -> n
})


def script_of_char(ch: str) -> str | None:
    cp = ord(ch)
    if INDIC_FIRST <= cp <= INDIC_LAST:
        return BLOCK_SCRIPTS[(cp - INDIC_FIRST) >> 7]
    return None


def detect_script(text: str) -> int:
    """SCRIPT_CODES value for one field: the Indic script with the most characters if
    any Indic character is present, else other (a non-Latin letter), else latin, else none."""
    counts: dict[str, int] = {}
    for ch in text:
        s = script_of_char(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1
    if counts:
        return SCRIPT_CODES[max(counts, key=counts.get)]
    if _OTHER.search(text):
        return SCRIPT_CODES["other"]
    if _LATIN.search(text):
        return SCRIPT_CODES["latin"]
    return SCRIPT_CODES["none"]


def romanise_run(run: str) -> tuple[str, int, int]:
    """One Indic run -> (ascii romanisation, script code, Indic char count). Uncached."""
    counts: dict[str, int] = {}
    dev = []
    for ch in run:
        s = script_of_char(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1
        for c in _PRE.get(ch, ch):
            if script_of_char(c) is None:
                dev.append(c)
                continue
            shifted = chr(ord(c) - ((ord(c) - INDIC_FIRST) & ~0x7F))
            dev.append(_PRE.get(shifted, shifted))
    script = max(counts, key=counts.get) if counts else "devanagari"
    word = "".join(dev)
    if script in SCHWA_DELETING and word:
        last = ord(word[-1])
        if last in _CONSONANT or (last == _NUKTA and len(word) > 1 and ord(word[-2]) in _CONSONANT):
            word += _VIRAMA
    out = sanscript.transliterate(word, sanscript.DEVANAGARI, sanscript.IAST)
    for pat, rep in _IAST_SUBS:
        out = pat.sub(rep, out)
    out = out.translate(_IAST_CHARS)
    out = "".join(c for c in out if not INDIC_FIRST <= ord(c) <= INDIC_LAST)  # anything the library left unmapped
    return out.lower(), SCRIPT_CODES[script], sum(counts.values())


class Romaniser:
    """Romanises text run by run with a per-instance memo. Only Indic runs are touched;
    everything else in the string passes through unchanged."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, int, int]] = {}
        self.hits = 0
        self.misses = 0
        self.fields = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def romanise_with_script(self, text: str) -> tuple[str, int]:
        self.fields += 1
        counts: dict[int, int] = {}

        def sub(m: regex.Match) -> str:
            run = m.group()
            hit = self._cache.get(run)
            if hit is None:
                self.misses += 1
                hit = self._cache[run] = romanise_run(run)
            else:
                self.hits += 1
            roman, code, n = hit
            counts[code] = counts.get(code, 0) + n
            return roman

        out = _INDIC_RUN.sub(sub, text)
        return out, (max(counts, key=counts.get) if counts else detect_script(text))

    def romanise(self, text: str) -> str:
        return self.romanise_with_script(text)[0]
