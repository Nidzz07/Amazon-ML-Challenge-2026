"""Text normalisation (Track C). Pure: no I/O, no module-level mutable state.

The work is written as polars expression builders so s1 runs it vectorised over
millions of rows. The string-level helpers (clean_text, accent_forms,
strip_legal_suffix, canon_street, normalise_record) run the SAME expressions on a
one-row frame, so tests and features see exactly what s1 produces.

Field pipeline:
  *_norm   NFKC -> lowercase -> junk tokens removed -> whitespace collapsed.
           Original script, accents and punctuation kept (a leading '"' survives).
  *_roman  *_norm -> Indic runs romanised (translit.Romaniser, only on rows that
           contain Indic characters) -> accents folded -> '&' -> 'and' ->
           punctuation to space -> [name: legal suffix stripped | addr: street
           types canonicalised] -> phonetic fold (doubled vowels collapsed,
           'ph' -> 'f'). The fold runs on every row, Latin and romanised alike, so
           both sides of a pair land on the same spelling (फूड -> phud -> fud,
           food -> fod... -> both reach the same n-grams as far as spelling allows).
The accent-stripped form is *_roman and the original is *_norm, so both are kept.
In addr_roman, hyphen/slash-joined house numbers ("8-2-293/82/c/16/a") stay one
token: splitting them on punctuation turns one distinctive token into several common
ones and costs the address TF-IDF channel its best evidence.

name_roman has its legal suffix stripped (fall back to the unstripped name if that
would empty it); the removed suffix is kept in name_suffix ('' if none).

Address components (parse_address_components) are read from the romanised address
with its punctuation intact, split on commas:
  street_num   first house-number token, verbatim (lowercase), never space-split
  postcode     a country-shaped code standing as its own component (see POSTCODE)
  state_canon  lowercase code from STATES (full names, codes, native-script
               romanisations); an unmapped 2-letter component passes through as is
  city_norm    the city-like component nearest the state (or, with no state, the
               street): no digits, not a state, no structural words (road, floor,
               near, c/o ...). Normalised like *_roman, never split further.
"""
import polars as pl

import translit

# --- cleaning -----------------------------------------------------------------------

JUNK_TOKENS = ("null", "n/a", "nil", "--", "-")
_SEP = r"[\s,;]"
_JUNK_RE = rf"(^|{_SEP})(?:{'|'.join(JUNK_TOKENS)})({_SEP}|$)"
_JUNK_ANYWHERE = r"<<|\(\s*\)"

# Accent folding: characters NFKD does not decompose, then combining marks dropped.
_FOLD_FROM = ["œ", "æ", "ß", "ø", "ł", "đ", "ð", "þ", "ı"]
_FOLD_TO = ["oe", "ae", "ss", "o", "l", "d", "d", "th", "i"]

# --- lexicons (hand-written) ----------------------------------------------------------

# Written in the post-cleaning form: lowercase, punctuation already turned into spaces
# ("L.L.C." -> "l l c", "& Co" -> "and co"), before doubled vowels are collapsed.
LEGAL_SUFFIXES = {
    "US": ("inc", "llc", "corp", "ltd", "plc", "lp", "llp", "incorporated", "corporation", "limited",
           "l l c", "l l p"),
    "India": ("pvt", "private", "ltd", "limited", "llp", "and co",
              # romanised Indic spellings of the same words (translit output)
              "praivet", "praibhet", "piraivet", "praivatt", "limitet", "limatid", "limittad", "pra li",
              "elelpi", "elaelapi"),
    "France": ("sarl", "sas", "sasu", "sa", "sci", "eurl", "snc"),
}
# Countries whose suffix may also lead the name ("SARL Ehpad Club").
LEADING_SUFFIX_COUNTRIES = ("France",)

STREET_TYPES = {
    "street": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "bd": "blvd", "lane": "ln", "drive": "dr", "court": "ct",
    "square": "sq", "highway": "hwy", "suite": "ste", "apartment": "apt", "floor": "fl",
}
_PO_BOX = r"\b(?:p\s?o|post)\s+box\b"

# State lookup: canonical lowercase code -> every spelling seen for it. Spellings are
# written naturally and folded through the same expressions as the addresses at
# lookup time. India's native-script entries are the romanised forms of the state
# names that occur in the data (महाराष्ट्र -> "maharashtr", दिल्ली -> "dilli", ...).
STATES = {
    "India": {
        "ap": ("andhra pradesh", "andhrapradesh", "ap"), "ar": ("arunachal pradesh", "ar"),
        "as": ("assam", "as"), "br": ("bihar", "br"),
        "cg": ("chhattisgarh", "chattisgarh", "chhatisgarh", "cg", "ct"), "ga": ("goa", "ga"),
        "gj": ("gujarat", "gj"), "hr": ("haryana", "hariyana", "hr"), "hp": ("himachal pradesh", "hp"),
        "jh": ("jharkhand", "jh"), "ka": ("karnataka", "karnatak", "ka"),
        "kl": ("kerala", "keralam", "keralan", "kl"), "mp": ("madhya pradesh", "madhy pradesh", "mp"),
        "mh": ("maharashtra", "maharashtr", "mh"), "mn": ("manipur", "mn"), "ml": ("meghalaya", "ml"),
        "mz": ("mizoram", "mz"), "nl": ("nagaland", "nl"),
        "od": ("odisha", "orissa", "orisha", "od", "or"), "pb": ("punjab", "panjab", "pb"),
        "rj": ("rajasthan", "rj"), "sk": ("sikkim", "sk"),
        "tn": ("tamil nadu", "tamilnadu", "tamilnatu", "tn"), "tg": ("telangana", "tg", "ts"),
        "tr": ("tripura", "tr"), "up": ("uttar pradesh", "up"),
        "uk": ("uttarakhand", "uttaranchal", "uk", "ut"),
        "wb": ("west bengal", "pashchimabang", "paschim banga", "wb"),
        "an": ("andaman and nicobar islands", "andaman and nicobar", "an"), "ch": ("chandigarh", "ch"),
        "dh": ("dadra and nagar haveli and daman and diu", "dadra and nagar haveli", "daman and diu", "dh", "dn", "dd"),
        "dl": ("delhi", "dilli", "nct of delhi", "dl"), "jk": ("jammu and kashmir", "jammu kashmir", "jk"),
        "la": ("ladakh", "la"), "ld": ("lakshadweep", "ld"), "py": ("puducherry", "pondicherry", "py"),
    },
    "US": {
        code: (name, code) for code, name in (
            ("al", "alabama"), ("ak", "alaska"), ("az", "arizona"), ("ar", "arkansas"), ("ca", "california"),
            ("co", "colorado"), ("ct", "connecticut"), ("de", "delaware"), ("dc", "district of columbia"),
            ("fl", "florida"), ("ga", "georgia"), ("hi", "hawaii"), ("id", "idaho"), ("il", "illinois"),
            ("in", "indiana"), ("ia", "iowa"), ("ks", "kansas"), ("ky", "kentucky"), ("la", "louisiana"),
            ("me", "maine"), ("md", "maryland"), ("ma", "massachusetts"), ("mi", "michigan"),
            ("mn", "minnesota"), ("ms", "mississippi"), ("mo", "missouri"), ("mt", "montana"),
            ("ne", "nebraska"), ("nv", "nevada"), ("nh", "new hampshire"), ("nj", "new jersey"),
            ("nm", "new mexico"), ("ny", "new york"), ("nc", "north carolina"), ("nd", "north dakota"),
            ("oh", "ohio"), ("ok", "oklahoma"), ("or", "oregon"), ("pa", "pennsylvania"),
            ("ri", "rhode island"), ("sc", "south carolina"), ("sd", "south dakota"), ("tn", "tennessee"),
            ("tx", "texas"), ("ut", "utah"), ("vt", "vermont"), ("va", "virginia"), ("wa", "washington"),
            ("wv", "west virginia"), ("wi", "wisconsin"), ("wy", "wyoming"), ("pr", "puerto rico"),
            ("gu", "guam"), ("vi", "virgin islands"),
        )
    },
}
# Countries where a lone unmapped 2-letter component is taken as the state, as is.
STATE_CODE_COUNTRIES = ("India", "US")

# Words that mark a component as street/building/landmark detail, never a city.
ADDRESS_STRUCTURE_WORDS = (
    "st", "street", "rd", "road", "ave", "avenue", "blvd", "boulevard", "ln", "lane", "dr", "drive", "ct",
    "court", "sq", "hwy", "highway", "ste", "suite", "apt", "apartment", "apartments", "fl", "floor",
    "pobox", "unit", "bldg", "building", "room", "tower", "wing", "shop", "complex", "plot", "flat",
    "house", "no", "block", "sector", "sec", "ward", "gali", "marg", "main", "cross", "near", "nr",
    "opp", "opposite", "behind", "beside", "c o", "residency", "society", "chs", "layout", "colony",
    "phase", "village", "vill", "post", "po", "dist", "district", "taluka", "tal", "tehsil", "mandal",
    "rue", "r", "av", "bd", "chemin", "allee", "place", "impasse", "route", "bp", "cs", "cedex", "bis",
)

# Postcode, matched against a whole folded component (punctuation already spaces).
#   US      optional state, then ZIP or ZIP+4 ("il 62701"). 5-digit numbers leading a
#           street ("20863 rebecca ln") or after "unit" are house/unit numbers.
#   India   optional place name, then a 6-digit PIN that cannot start with 0.
#   France  5 digits with a real department prefix (01-98), optionally followed by
#           the town ("33000 bordeaux"). "00133 r croix" is a house number.
POSTCODE = {
    "US": r"^(?:[a-z ]+ )?(\d{5})(?: \d{4})?$",
    "India": r"^(?:[a-z ]+ )?([1-9]\d{5})$",
    "France": r"^((?:0[1-9]|[1-8]\d|9[0-8])\d{3})(?: [a-z][a-z ]*)?$",
}

# House number: hyphen/slash-joined alphanumeric segments with at least one digit.
_HOUSE_NUMBER = r"(?:[0-9a-z]+[-/])*[0-9a-z]*[0-9][0-9a-z]*(?:[-/][0-9a-z]+)*"
_HOUSE_NUMBER_TOKEN = rf"(?:^|[^0-9a-z/\-]){_HOUSE_NUMBER}"
_ORDINAL = r"^\d+(?:st|nd|rd|th)$"
# Placeholders that carry '-' and '/' through punctuation-to-space when a digit sits
# on either side ("b-1/313-e"), so addr_roman keeps house numbers whole.
_JOIN_PLACEHOLDERS = (("-", ""), ("/", ""))


def clean_expr(e: pl.Expr) -> pl.Expr:
    e = e.fill_null("").str.normalize("NFKC").str.to_lowercase()
    e = e.str.replace_all(_JUNK_ANYWHERE, " ")
    for _ in range(2):  # adjacent junk tokens share a separator, so one pass misses every other one
        e = e.str.replace_all(_JUNK_RE, "${1} ${2}")
    return (
        e.str.replace_all(r"\s+", " ")
        .str.replace_all(r" ([,;])", "${1}")
        .str.replace_all(r"([,;])(?: ?[,;])+", "${1}")
        .str.strip_chars(" ,;")
    )


def fold_accents_expr(e: pl.Expr) -> pl.Expr:
    return e.str.replace_many(_FOLD_FROM, _FOLD_TO).str.normalize("NFKD").str.replace_all(r"\p{Mn}", "")


def roman_base_expr(e: pl.Expr, keep_house_numbers: bool = False) -> pl.Expr:
    e = fold_accents_expr(e).str.replace_all("&", " and ")
    if keep_house_numbers:
        for _ in range(2):  # "8-2-293": adjacent joiners share a digit, so one pass misses every other one
            for joiner, ph in _JOIN_PLACEHOLDERS:
                e = e.str.replace_all(rf"(\d){joiner}([0-9a-z])", f"${{1}}{ph}${{2}}")
                e = e.str.replace_all(rf"([0-9a-z]){joiner}(\d)", f"${{1}}{ph}${{2}}")
    kept = "".join(ph for _, ph in _JOIN_PLACEHOLDERS) if keep_house_numbers else ""
    e = e.str.replace_all(rf"[^\p{{L}}\p{{N}}{kept}]+", " ").str.strip_chars()
    if keep_house_numbers:
        for joiner, ph in _JOIN_PLACEHOLDERS:
            e = e.str.replace_all(ph, joiner, literal=True)
    return e


def phonetic_fold_expr(e: pl.Expr) -> pl.Expr:
    """Doubled vowels collapsed, and 'ph' -> 'f': Indic फ romanises as 'ph' but is 'f'
    in the English words that make up most Indian business names (फूड = food)."""
    for v in "aeiou":
        e = e.str.replace_all(f"{v}{{2,}}", v)
    return e.str.replace_all("ph", "f")


def _alts(words) -> str:
    return "|".join(w.replace(" ", r"\s") for w in sorted(words, key=len, reverse=True))


def legal_suffix_exprs(name: pl.Expr, country: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """(stripped, removed) for a roman-base name. removed is the stripped suffix text
    ('' if none). A name that is nothing but suffixes is left whole."""
    stripped, removed = name, pl.lit("")
    for c, words in LEGAL_SUFFIXES.items():
        alts = _alts(words)
        trail = rf"(?:^|\s)((?:{alts})(?:\s(?:{alts}))*)$"
        lead = rf"^((?:{alts})(?:\s(?:{alts}))*)(?:\s|$)"
        s = name.str.replace(trail, "")
        r = name.str.extract(trail, 1).fill_null("")
        if c in LEADING_SUFFIX_COUNTRIES:
            r = pl.concat_str([s.str.extract(lead, 1).fill_null(""), r], separator=" ").str.strip_chars()
            s = s.str.replace(lead, "")
        s = s.str.strip_chars()
        keep = s == ""
        stripped = pl.when(country == c).then(pl.when(keep).then(name).otherwise(s)).otherwise(stripped)
        removed = pl.when(country == c).then(pl.when(keep).then(pl.lit("")).otherwise(r)).otherwise(removed)
    return stripped, removed


def street_expr(addr: pl.Expr) -> pl.Expr:
    return (
        addr.str.replace_all(_PO_BOX, "pobox")
        .str.split(" ")
        .list.eval(pl.element().replace(STREET_TYPES))
        .list.join(" ")
    )


def tokens_expr(e: pl.Expr) -> pl.Expr:
    return e.str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


# --- address components -----------------------------------------------------------------

def _fold_words(words) -> list[str]:
    """Lexicon spellings -> the folded form addresses are compared in."""
    return (
        pl.DataFrame({"w": list(words)})
        .select(phonetic_fold_expr(roman_base_expr(pl.col("w"))))
        .to_series()
        .to_list()
    )


def _state_table() -> pl.DataFrame:
    rows = [(country, code, spelling) for country, codes in STATES.items()
            for code, spellings in codes.items() for spelling in spellings]
    df = pl.DataFrame(rows, schema=["country", "state", "alpha"], orient="row")
    return df.with_columns(pl.Series("alpha", _fold_words(df["alpha"]))).unique(["country", "alpha"], keep="first")


def parse_address_components(df: pl.DataFrame) -> pl.DataFrame:
    """(addr_pre, country) -> (street_num, city_norm, state_canon, postcode), same row
    order, '' where absent. addr_pre is the cleaned address after romanisation, with
    its punctuation intact. Nothing is ever dropped: an address that parses to nothing
    yields four empty strings."""
    rows = df.select(pl.col("addr_pre").fill_null(""), pl.col("country").fill_null("")).with_row_index("row")
    stop = "|".join(sorted({w.replace(" ", r"\s") for w in _fold_words(ADDRESS_STRUCTURE_WORDS)}, key=len, reverse=True))

    comps = (
        rows.select("row", "country", pl.col("addr_pre").str.split(",").alias("comp"))
        .explode("comp", empty_as_null=False)
        .with_columns(pl.int_range(pl.len()).over("row").alias("pos"))
        .with_columns(phonetic_fold_expr(roman_base_expr(pl.col("comp"))).alias("key"))
        .with_columns(
            # the component with every digit-bearing token removed: "delhi 110092" -> "delhi"
            pl.col("key").str.replace_all(r"\S*\d\S*", " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("alpha"),
            pl.col("key").str.contains(r"\d").alias("has_digit"),
        )
        .join(_state_table(), on=["country", "alpha"], how="left")
    )
    pc_value = pl.lit(None, dtype=pl.String)
    for country, pat in POSTCODE.items():
        hit = (pl.col("country") == country) & pl.col("key").str.contains(pat)
        if country == "US":  # the prefix must be a state, and a lone ZIP never leads the address
            hit &= pl.col("state").is_not_null() | ((pl.col("alpha") == "") & (pl.col("pos") > 0))
        if country == "France":
            hit &= ~pl.col("key").str.contains(rf"\b(?:{stop})\b")
        pc_value = pl.when(hit).then(pl.col("key").str.extract(pat, 1)).otherwise(pc_value)
    comps = comps.with_columns(pc_value.alias("pc"))

    # State: the first mapped component; failing that, a lone 2-letter component, as is.
    code_like = pl.col("country").is_in(STATE_CODE_COUNTRIES) & pl.col("alpha").str.contains(r"^[a-z]{2}$") & ~pl.col("has_digit")
    per_row = comps.sort("row", "pos").group_by("row", maintain_order=True).agg(
        pl.col("pc").drop_nulls().first().alias("postcode"),
        pl.col("state").drop_nulls().first().alias("mapped_state"),
        pl.col("pos").filter(pl.col("state").is_not_null()).first().alias("mapped_pos"),
        pl.col("alpha").filter(code_like).first().alias("code_state"),
        pl.col("pos").filter(code_like).first().alias("code_pos"),
        pl.col("pos").filter(pl.col("has_digit") & pl.col("pc").is_null()).first().alias("street_pos"),
    ).with_columns(
        pl.coalesce("mapped_state", "code_state").alias("state_canon"),
        pl.coalesce("mapped_pos", "code_pos").alias("state_pos"),
    )

    # City: nearest city-like component to the state, else to the street component.
    city_text = (
        pl.when(~pl.col("has_digit")).then(pl.col("key"))
        .when(pl.col("pc").is_not_null()).then(pl.col("alpha"))  # "33000 bordeaux" -> "bordeaux"
    )
    anchor = pl.coalesce("state_pos", "street_pos", pl.lit(0))
    city = (
        comps.join(per_row.select("row", "state_pos", "street_pos"), on="row", how="left")
        .with_columns(city_text.alias("city"))
        .filter(
            pl.col("city").is_not_null() & (pl.col("city") != "")
            & pl.col("state").is_null()
            & (pl.col("pos") != pl.col("state_pos").fill_null(-1))
            & ~pl.col("city").str.contains(rf"\b(?:{stop})\b")
            & (pl.col("city").str.count_matches(" ") < 4)
        )
        .with_columns((pl.col("pos") - anchor).abs().alias("dist"))
        .sort("row", "dist", "pos")
        .group_by("row", maintain_order=True)
        .agg(pl.col("city").first().alias("city_norm"))
    )

    # Street number: first house-number token of the whole address, verbatim, never the
    # postcode and never a bare ordinal ("1st main").
    street = (
        rows.join(per_row.select("row", "postcode"), on="row", how="left")
        .select("row", "postcode", pl.col("addr_pre").str.extract_all(_HOUSE_NUMBER_TOKEN).alias("hn"))
        .explode("hn", empty_as_null=True)
        .with_columns(pl.col("hn").str.replace(r"^[^0-9a-z]+", ""))
        .filter(pl.col("hn").is_not_null() & ~pl.col("hn").str.contains(_ORDINAL)
                & (pl.col("hn") != pl.col("postcode").fill_null("")))
        .group_by("row", maintain_order=True)
        .agg(pl.col("hn").first().alias("street_num"))
    )
    return (
        rows.select("row")
        .join(street, on="row", how="left")
        .join(city, on="row", how="left")
        .join(per_row.select("row", "state_canon", "postcode"), on="row", how="left")
        .sort("row")
        .select(pl.col("street_num", "city_norm", "state_canon", "postcode").fill_null(""))
    )


# --- romanisation hook ------------------------------------------------------------------

def romanise_series(s: pl.Series, romaniser: translit.Romaniser) -> tuple[pl.Series, pl.Series]:
    """(text with Indic runs romanised, per-field script code). Rows with no Indic
    character never reach Python: the regex pre-check routes them past the romaniser."""
    df = s.to_frame("t")
    indic = df.select(pl.col("t").str.contains(translit.INDIC_PATTERN)).to_series()
    codes = df.select(
        pl.when(pl.col("t").str.contains(translit.OTHER_SCRIPT_PATTERN)).then(translit.SCRIPT_CODES["other"])
        .when(pl.col("t").str.contains(translit.LATIN_PATTERN)).then(translit.SCRIPT_CODES["latin"])
        .otherwise(translit.SCRIPT_CODES["none"]).cast(pl.UInt8)
    ).to_series()
    idx = indic.arg_true()
    if len(idx) == 0:
        return s, codes
    out = [romaniser.romanise_with_script(t) for t in s.gather(idx).to_list()]
    roman = s.clone().scatter(idx, [o[0] for o in out])
    codes = codes.clone().scatter(idx, [o[1] for o in out])
    return roman, codes


# --- frame -------------------------------------------------------------------------------

def normalise_frame(df: pl.DataFrame, romaniser: translit.Romaniser) -> pl.DataFrame:
    """records schema (entity_id, business_name, business_address, country) -> norm schema,
    row order preserved."""
    base = df.select(
        "entity_id",
        clean_expr(pl.col("business_name")).alias("name_norm"),
        clean_expr(pl.col("business_address")).alias("addr_norm"),
        pl.col("country").fill_null(""),
    )
    name_pre, name_script = romanise_series(base["name_norm"], romaniser)
    addr_pre, addr_script = romanise_series(base["addr_norm"], romaniser)
    parts = parse_address_components(pl.DataFrame({"addr_pre": addr_pre, "country": base["country"]}))
    stripped, removed = legal_suffix_exprs(pl.col("name_base"), pl.col("country"))
    return (
        base.with_columns(
            roman_base_expr(pl.lit(name_pre)).alias("name_base"),
            roman_base_expr(pl.lit(addr_pre), keep_house_numbers=True).alias("addr_base"),
            # name script in the low nibble, address script in the high nibble
            (pl.lit(name_script).cast(pl.UInt16) + pl.lit(addr_script).cast(pl.UInt16) * 16).cast(pl.UInt8).alias("script"),
        )
        .with_columns(
            phonetic_fold_expr(stripped).alias("name_roman"),
            removed.alias("name_suffix"),
            phonetic_fold_expr(street_expr(pl.col("addr_base"))).alias("addr_roman"),
        )
        .with_columns(parts)
        .with_columns(
            tokens_expr(pl.col("name_roman")).alias("name_tokens"),
            tokens_expr(pl.col("addr_roman")).alias("addr_tokens"),
        )
        .select(
            "entity_id",
            "name_norm",
            "name_roman",
            "name_tokens",
            pl.col("name_tokens").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("name_acronym"),
            "addr_norm",
            "addr_roman",
            "addr_tokens",
            "street_num",
            "city_norm",
            "state_canon",
            "postcode",
            "country",
            (pl.col("addr_norm") != "").alias("has_addr"),
            "script",
            "name_suffix",
        )
    )


# --- string-level helpers (same expressions, one row) -------------------------------------

def _one(expr: pl.Expr, **cols) -> object:
    return pl.DataFrame({k: [v] for k, v in cols.items()}, schema={k: pl.String for k in cols}).select(expr).item()


def clean_text(s: str) -> str:
    return _one(clean_expr(pl.col("s")), s=s)


def fold_accents(s: str) -> str:
    return _one(fold_accents_expr(pl.col("s")), s=s)


def accent_forms(s: str) -> tuple[str, str]:
    """(original, accent-stripped), both cleaned."""
    c = clean_text(s)
    return c, fold_accents(c)


def strip_legal_suffix(name: str, country: str) -> tuple[str, str]:
    """(stripped, removed) for a roman-base name (lowercase, punctuation already spaces)."""
    stripped, removed = legal_suffix_exprs(pl.col("n"), pl.col("c"))
    out = pl.DataFrame({"n": [name], "c": [country]}).select(stripped.alias("s"), removed.alias("r"))
    return out["s"][0], out["r"][0]


def canon_street(addr: str) -> str:
    return _one(street_expr(pl.col("s")), s=addr)


def normalise_record(name: str, address: str, country: str, romaniser: translit.Romaniser | None = None) -> dict:
    df = pl.DataFrame({"entity_id": ["x"], "business_name": [name], "business_address": [address], "country": [country]})
    return normalise_frame(df, romaniser or translit.Romaniser()).row(0, named=True)
