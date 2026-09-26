"""Acceptance + unit tests for normalise.py / translit.py (Track C).

The pair tests read tests/fixtures/normalise_pairs.tsv (regenerate with
tests/fixtures/make_normalise_fixture.py): real true-match pairs from the train
ground truth, plus non-matching negatives. Similarity is rapidfuzz token_set_ratio
on the normalised field the blocking channels consume (name_roman / addr_roman).
"""
from pathlib import Path

import polars as pl
import pytest
from rapidfuzz import fuzz

import config
import normalise
import pipeline_io as pio
import translit

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "normalise_pairs.tsv"
CROSS_SCRIPT_FLOOR = 0.60  # romanised Indic vs Latin: fuzzy matching closes the rest
LATIN_FLOOR = 0.80
NEGATIVE_CEIL = 0.60

# Known gaps, keyed by fixture ids. strict=True: if one starts passing, the test fails
# so the entry gets removed.
KNOWN_GAPS = {
    "S1-645483272|S3-128774057": "ऑल फूड -> 'ol fud' vs 'all fod': short names, vowel quality lost",
    "S1-986269970|S2-580298892": "Oriya medial schwa: 'kanashtrakasan' vs 'construction'",
    "S1-835448813|S2-42826158": "Tamil has no voiced stops or /s/-/ch/ split: 'tirim chistams' vs 'dream systems'",
    "S1-833469626|S2-951607048": "Sree/Shri spelling variants (needs the deferred mined lexicon)",
    "S1-386020350|S3-135072481": "NC vs North Carolina: state canonicalisation is out of this scope",
}

CASES = pl.read_csv(FIXTURE, separator=config.TSV_SEP, quote_char=None, infer_schema=False,
                    encoding="utf8").to_dicts()


def sim(a: str, b: str) -> float:
    assert a and b, "normalisation must not empty a real field"
    return fuzz.token_set_ratio(a, b) / 100


def roman(case: dict, side: str, rom: translit.Romaniser) -> str:
    text = case[side]
    if case["field"] == "name":
        return normalise.normalise_record(text, "", case["country"], rom)["name_roman"]
    return normalise.normalise_record("x", text, case["country"], rom)["addr_roman"]


@pytest.fixture(scope="module")
def rom():
    return translit.Romaniser()


def test_known_gaps_exist_in_fixture():
    ids = {c["ids"] for c in CASES}
    assert set(KNOWN_GAPS) <= ids


def test_fixture_shape():
    cats = pl.DataFrame(CASES)["category"].value_counts()
    counts = dict(zip(cats["category"], cats["count"]))
    assert counts["cross_script"] >= 40
    assert sum(counts[c] for c in ("typo", "accent", "abbrev", "reorder")) >= 20
    assert counts["negative"] >= 10
    assert any(c["left"].startswith('"') for c in CASES), "need a name starting with a literal quote"


def _positive(c: dict):
    gap = KNOWN_GAPS.get(c["ids"])
    marks = [pytest.mark.xfail(reason=gap, strict=True)] if gap else []
    return pytest.param(c, marks=marks, id=f"{c['case_id']}-{c['category']}")


@pytest.mark.parametrize("case", [_positive(c) for c in CASES if c["category"] != "negative"])
def test_positive_pair_above_floor(case, rom):
    floor = CROSS_SCRIPT_FLOOR if case["category"] == "cross_script" else LATIN_FLOOR
    a, b = roman(case, "left", rom), roman(case, "right", rom)
    assert sim(a, b) >= floor, f"{case['left']!r} -> {a!r} vs {case['right']!r} -> {b!r}: {sim(a, b):.2f} < {floor}"


@pytest.mark.parametrize("case", [c for c in CASES if c["category"] == "negative"], ids=lambda c: f"{c['case_id']}-negative")
def test_negative_pair_below_ceiling(case, rom):
    a, b = roman(case, "left", rom), roman(case, "right", rom)
    assert sim(a, b) < NEGATIVE_CEIL, f"{a!r} vs {b!r}: {sim(a, b):.2f} >= {NEGATIVE_CEIL}"


def test_cross_script_romanised_output_is_ascii(rom):
    for c in CASES:
        if c["category"] == "cross_script":
            out = roman(c, "right", rom)
            assert out.isascii(), f"{c['right']!r} -> {out!r}"


# --- literal quote ----------------------------------------------------------------

def test_leading_quote_is_not_noise(rom):
    rec = normalise.normalise_record('"ehpad Club SAS', "", "France", rom)
    assert rec["name_norm"].startswith('"'), "name_norm keeps the original characters"
    assert rec["name_roman"] == "ehpad club"
    assert rec["name_tokens"] == ["ehpad", "club"]


# --- normalise.py unit tests --------------------------------------------------------

def test_clean_text_nfkc_lower_whitespace():
    assert normalise.clean_text("  ＡＣＭＥ   Ｉｎｃ  ") == "acme inc"


@pytest.mark.parametrize("raw,want", [
    ("CEDAR STREET, null, MONTICELLO, IA", "cedar street, monticello, ia"),
    ("Fairfield, Bldg N/A, 138 Armstrong Street", "fairfield, bldg, 138 armstrong street"),
    ("-- Sulphur Clinic Co", "sulphur clinic co"),
    ("<< Team Ecole", "team ecole"),
    ("FN () Private Limited", "fn private limited"),
    ("NO 186 1 NIL JAYPUR", "no 186 1 jaypur"),
    ("Jalandhar - I Communications", "jalandhar i communications"),
    ("N0032 N/A N/A K N Roy Road", "n0032 k n roy road"),
    ("null", ""),
])
def test_junk_tokens_removed(raw, want):
    assert normalise.clean_text(raw) == want


def test_accent_forms_keep_both():
    original, stripped = normalise.accent_forms("Énterprises Bóral Œuvre")
    assert original == "énterprises bóral œuvre"
    assert stripped == "enterprises boral oeuvre"


@pytest.mark.parametrize("name,country,stripped,removed", [
    ("raj investments llp", "India", "raj investments", "llp"),
    ("ss food private limited", "India", "ss food", "private limited"),
    ("acme pvt ltd", "India", "acme", "pvt ltd"),
    ("sharma and co", "India", "sharma", "and co"),
    ("boyd newtekone llc", "US", "boyd newtekone", "llc"),
    ("stag inc", "US", "stag", "inc"),
    ("znb club sarl", "France", "znb club", "sarl"),
    ("sarl ehpad club", "France", "ehpad club", "sarl"),
    ("private limited", "India", "private limited", ""),  # never strip a name to nothing
    ("acme sarl", "US", "acme sarl", ""),  # suffixes are per country
])
def test_legal_suffix(name, country, stripped, removed):
    assert normalise.strip_legal_suffix(name, country) == (stripped, removed)


@pytest.mark.parametrize("raw,want", [
    ("1536 biggs street amarillo tx", "1536 biggs st amarillo tx"),
    ("617 rock hill church road", "617 rock hill church rd"),
    ("12 saint marks avenue", "12 st marks ave"),
    ("9 sunset boulevard suite 4 floor 2", "9 sunset blvd ste 4 fl 2"),
    ("po box 77 pine lane", "pobox 77 pine ln"),
    ("p o box 5 main drive apartment 3", "pobox 5 main dr apt 3"),
    ("1 king court highway 9 square", "1 king ct hwy 9 sq"),
])
def test_street_abbreviations(raw, want):
    assert normalise.canon_street(raw) == want


def test_normalise_frame_schema_and_order(rom):
    df = pl.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-3"],
        "business_name": ["Raj Investments LLP", "विजय प्रोजेक्ट्स प्राइवेट लिमिटेड", '"ehpad Club SAS'],
        "business_address": ["12, MG Road, null, महाराष्ट्र", "", "4 Rue Daurat, Saint-Nazaire"],
        "country": ["India", "India", "France"],
    })
    out = normalise.normalise_frame(df, rom)
    pio.check_schema(out, pio.NORM_SCHEMA, "normalise_frame")
    assert out["entity_id"].to_list() == ["S1-1", "S1-2", "S1-3"]
    assert out["has_addr"].to_list() == [True, False, True]
    assert out["street_num"][0] == "12"
    assert out["name_acronym"][0] == "ri"


# --- translit.py unit tests ---------------------------------------------------------

@pytest.mark.parametrize("text,script", [
    ("", "none"), ("123 - ,", "none"), ("Acme Inc", "latin"), ("Énterprises", "latin"),
    ("विजय", "devanagari"), ("রাম", "bengali"), ("ਸਿਲਵਰ", "gurmukhi"), ("શિવમ", "gujarati"),
    ("ଶିବ", "oriya"), ("ஜைன்", "tamil"), ("గురు", "telugu"), ("ರಾಮ್", "kannada"), ("ശക്തി", "malayalam"),
    ("Smart नॉर्थ कंसल्टेंट्स Private Limited", "devanagari"),  # any Indic run wins over Latin
    ("Москва", "other"),
])
def test_detect_script(text, script):
    assert translit.detect_script(text) == translit.SCRIPT_CODES[script]


def test_romanise_samples(rom):
    assert rom.romanise("विजय प्रोजेक्ट्स प्राइवेट लिमिटेड") == "vijay projekts praivet limited"
    assert rom.romanise("ரா") == "ra"
    assert rom.romanise("Acme 12") == "Acme 12"  # non-Indic text passes through untouched


def test_token_cache_counts():
    r = translit.Romaniser()
    r.romanise("प्राइवेट लिमिटेड")
    assert (r.hits, r.misses) == (0, 2)
    r.romanise("लिमिटेड प्राइवेट लिमिटेड")
    assert (r.hits, r.misses) == (3, 2)
    assert r.hit_rate == pytest.approx(3 / 5)


def test_latin_rows_skip_transliteration():
    r = translit.Romaniser()
    df = pl.DataFrame({
        "entity_id": ["S2-1", "S2-2"], "business_name": ["Acme Inc", "Énterprises SARL"],
        "business_address": ["1 Main St", "5 Rue X"], "country": ["US", "France"],
    })
    normalise.normalise_frame(df, r)
    assert r.fields == 0 and r.hits + r.misses == 0


# --- address parsing (parse_address_components) -------------------------------------
# Real addresses from the train/test files unless marked synthetic. Components are
# asserted in their normalised form (lowercase, accents folded).

def addr(address: str, country: str, rom=None) -> dict:
    return normalise.normalise_record("x", address, country, rom)


@pytest.mark.parametrize("address,street_num", [
    ("H.No.8-2-293/82/C/16/A, Road No.07, Jubilee Hills, Telangana, Plot No.16 3Rd Floor, Susheela Pride", "8-2-293/82/c/16/a"),
    ("D.No.20-6-211/1, Shah Ali Banda, Hyderabad, Telangana", "20-6-211/1"),
    ("H.No: 6-3-1238/B/21, Asif Avenue, Raj Bhavan Road, Hyderabad, Telangana", "6-3-1238/b/21"),
    ("B-32/304, Amrapali, Shantinagar Chs, Nr Tmt Bus Stop, Sec-11, Shantinagar, Mi, Ra Road(E), Thane, Maharashtra", "b-32/304"),
    ("B/2/83, Madhuvrund Chs Ltd, Nr. Rannapark, Ghatlodia, Ahmedabad, Gujarat", "b/2/83"),
    ("Medak, Telangana, Shivajinagar, Siddipet, 8-2-67/1/A/3/1", "8-2-67/1/a/3/1"),
    ("Plot No. S-5-A, Kartarpura, 22 Godown Industrial Area, Jaipur, Jaipur, Rajasthan", "s-5-a"),
    ("B-1/313-E Ground Floor Gali No.10, New Ashok Nagar, Delhi, East Delhi, Delhi", "b-1/313-e"),
    ("#216, 1St Main, Sapthagiri Residency, Muthurayanagar, Mysore Road, Bangalore, Karnataka", "216"),
    ("# 13-481 Sanjeevaiah Nagar, Beside I.B, Mancherial, Anantapur, Telangana", "13-481"),
])
def test_indian_house_number_is_atomic(address, street_num, rom):
    rec = addr(address, "India", rom)
    assert rec["street_num"] == street_num
    assert " " not in rec["street_num"]
    assert street_num in rec["addr_tokens"], "addr_roman must keep the house number as one token too"


@pytest.mark.parametrize("address,city,state", [
    ("D.No.20-6-211/1, Shah Ali Banda, Hyderabad, Telangana", "hyderabad", "tg"),
    ("H.No: 6-3-1238/B/21, Asif Avenue, Raj Bhavan Road, Hyderabad, Telangana", "hyderabad", "tg"),
    ("B/2/83, Madhuvrund Chs Ltd, Nr. Rannapark, Ghatlodia, Ahmedabad, Gujarat", "ahmedabad", "gj"),
    ("#216, 1St Main, Sapthagiri Residency, Muthurayanagar, Mysore Road, Bangalore, Karnataka", "bangalore", "ka"),
    ("FL NO 00302, SHIVANI RESIDENCY S NO 29, PUNE CITY, महाराष्ट्र", "pune city", "mh"),  # native-script state
    ("148, EAST DELHI, SHAHDARA, Delhi", "shahdara", "dl"),
    ("Mirzapur, Ews 12, Uttar Pradesh, Mirzapursadar, Awas Vikas Colony", None, "up"),
    ("Kolkata, 7, Brabourne Road, WB", "kolkata", "wb"),
    ("12, MG Road, Tamilnadu", "", "tn"),  # no city-like component: blank, not a street
])
def test_indian_city_and_state(address, city, state, rom):
    rec = addr(address, "India", rom)
    assert rec["state_canon"] == state
    if city is not None:
        assert rec["city_norm"] == city


@pytest.mark.parametrize("address,street_num,city,state,postcode", [
    ("2621 Cotten Road, Tyler, TX", "2621", "tyler", "tx", ""),
    ("IA, Iowa City, 1064 Newton Rd, Unit 11", "1064", "iowa city", "ia", ""),
    ("KANSAS CITY, MO, 630 45ND TERRACE, null", "630", "kansas city", "mo", ""),
    ("1536 Biggs St, Texas, Amarillo", "1536", "amarillo", "tx", ""),
    ("20863 REBECCA LANE, SAN BENITO, TX", "20863", "san benito", "tx", ""),  # leading 5 digits = house number
    ("273 Big Station Camp Boulevard, Unit 13103, Gallatin, TN", "273", "gallatin", "tn", ""),  # unit, not a ZIP
    ("TN, SODDY DAISY, 7607 DAYTON PIKE", "7607", "soddy daisy", "tn", ""),
    ("55 Main St, Springfield, IL 62701", "55", "springfield", "il", "62701"),  # synthetic: ZIPs are ~absent in the data
])
def test_us_address_components(address, street_num, city, state, postcode, rom):
    rec = addr(address, "US", rom)
    assert (rec["street_num"], rec["city_norm"], rec["state_canon"], rec["postcode"]) == (street_num, city, state, postcode)


@pytest.mark.parametrize("address,street_num,postcode", [
    ("14 RUE DIMU, 33000 BORDEAUX, BORDEAUX, Gironde", "14", "33000"),
    ("59430 DUNKERQUE, 11 R DU MARÉCHAL FOCH, DUNKERQUE", "11", "59430"),
    ("00133 R CROIX DE SEGUEY, BORDEAUX, Gironde", "00133", ""),  # 00xxx is not a French department
    ("4 Rue Daurat, Saint-Nazaire, Pays de la Loire", "4", ""),
])
def test_france_postcode_and_street(address, street_num, postcode, rom):
    rec = addr(address, "France", rom)
    assert (rec["street_num"], rec["postcode"]) == (street_num, postcode)


def test_unmapped_state_passes_through(rom):
    rec = addr("12 Foo Road, Springfield, ZZ", "US", rom)  # synthetic
    assert rec["state_canon"] == "zz" and rec["city_norm"] == "springfield"


def test_unparseable_address_keeps_row(rom):
    rec = addr("Near Old Bus Stand", "India", rom)
    assert (rec["street_num"], rec["state_canon"], rec["postcode"]) == ("", "", "")
    assert rec["has_addr"] is True


def test_parse_address_components_frame_order():
    df = pl.DataFrame({"addr_pre": ["2621 cotten road, tyler, tx", "", "d.no.20-6-211/1, hyderabad, telangana"],
                       "country": ["US", "India", "India"]})
    out = normalise.parse_address_components(df)
    assert out.columns == ["street_num", "city_norm", "state_canon", "postcode"]
    assert out["street_num"].to_list() == ["2621", "", "20-6-211/1"]
    assert out["state_canon"].to_list() == ["tx", "", "tg"]


def test_suffix_flag_is_persisted(rom):
    assert addr("x", "India", rom)["name_suffix"] == ""
    rec = normalise.normalise_record("Raj Investments LLP", "", "India", rom)
    assert (rec["name_roman"], rec["name_suffix"]) == ("raj investments", "llp")
    rec = normalise.normalise_record("SARL Ehpad Club", "", "France", rom)
    assert (rec["name_roman"], rec["name_suffix"]) == ("ehpad club", "sarl")
