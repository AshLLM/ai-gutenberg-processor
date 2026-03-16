"""Utilities for the Project Gutenberg pipeline.

Covers two concerns:
1. Metadata – scrape a Gutenberg ebook page and persist a JSON file.
2. Text processing – normalise, locate anchor points, and set up the OpenAI client.

Dependencies: requests, beautifulsoup4, openai, python-dotenv
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import find_dotenv, load_dotenv
import openai


# ---------------------------------------------------------------------------
# OpenAI setup
# ---------------------------------------------------------------------------


def setup_openai() -> openai.OpenAI:
    """Load OPENAI_KEY.env and return an initialised OpenAI client."""
    env_file = find_dotenv("OPENAI_KEY.env")
    if not env_file:
        raise FileNotFoundError(
            "Could not find 'OPENAI_KEY.env' in the current or parent directories."
        )
    load_dotenv(env_file)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment file")
    return openai.OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Metadata – private helpers
# ---------------------------------------------------------------------------


def _clean_text(s: Optional[str]) -> str:
    """Collapse whitespace in *s* and strip leading/trailing spaces."""
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def _extract_ebook_id(url: str) -> Optional[str]:
    """Pull the numeric ebook ID from a Gutenberg URL.

    Works for both page URLs (ebooks/11) and file URLs (/epub/11/).
    """
    match = re.search(r"ebooks/(\d+)|/epub/(\d+)(?:/|$)", url)
    if match:
        return match.group(1) or match.group(2)
    return None


YEAR_RANGE_PATTERN = re.compile(
    r"\b(?:published|written|composed|completed|created)"
    r"\s+(?:in\s+)?(?:the\s+)?"
    r"(\d{4})\s+to\s+(\d{4})\b",
    re.IGNORECASE,
)

SINGLE_YEAR_PATTERNS = [
    (re.compile(r"\bpublished\s+in\s+(\d{4})\b", re.IGNORECASE), "high"),
    (
        re.compile(
            r"\b(?:first|originally)\s+published\s+(?:in\s+)?(\d{4})\b",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        re.compile(r"\b(?:written|composed|completed)\s+in\s+(\d{4})\b", re.IGNORECASE),
        "high",
    ),
    (
        re.compile(
            r"\b(?:novel|poem|story|work|essay|play)\s+(?:of|in)\s+(\d{4})\b",
            re.IGNORECASE,
        ),
        "high",
    ),
    (re.compile(r"\((\d{4})\)", re.IGNORECASE), "low"),
]

TRANSLATION_PATTERNS = [
    (re.compile(r"\btranslated\s+in\s+(\d{4})\b", re.IGNORECASE), "high"),
    (
        re.compile(r"\btranslation\s+(?:published\s+)?in\s+(\d{4})\b", re.IGNORECASE),
        "high",
    ),
    (re.compile(r"\b(\d{4})\s+translation\b", re.IGNORECASE), "high"),
    (re.compile(r"\((\d{4})\)", re.IGNORECASE), "low"),
]

PLAUSIBLE_YEAR_RANGE = (400, 1950)


def _is_plausible(year: int) -> bool:
    return PLAUSIBLE_YEAR_RANGE[0] <= year <= PLAUSIBLE_YEAR_RANGE[1]


def _sanity_check_against_persons(
    year: int,
    persons: List[dict],
    posthumous_window: int = 10,
) -> bool:
    """Return True if *year* is plausible given available birth/death years."""
    for person in persons:
        birth = person.get("birth_year")
        death = person.get("death_year")
        if birth is not None and year < birth:
            return False
        if death is not None and year > death + posthumous_window:
            return False
    return True


def _extract_year_range(text: str) -> Optional[Tuple[int, int]]:
    match = YEAR_RANGE_PATTERN.search(text)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        if _is_plausible(start) and _is_plausible(end):
            return start, end
    return None


def _extract_single_year(
    text: str,
    patterns: List[Tuple[re.Pattern, str]],
    sanity_persons: List[dict],
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Return ``(year, source, confidence)`` or ``(None, None, None)``."""
    for pattern, confidence in patterns:
        match = pattern.search(text)
        if match:
            year = int(match.group(1))
            if _is_plausible(year) and _sanity_check_against_persons(
                year, sanity_persons
            ):
                return year, "summary_regex", confidence
    return None, None, None


def _extract_publication_dates(summaries: List[str], authors: List[dict]) -> dict:
    """Extract publication date fields for non-translated works."""
    text = " ".join(summaries)

    date_range = _extract_year_range(text)
    if date_range:
        start, end = date_range
        if _sanity_check_against_persons(
            start, authors
        ) and _sanity_check_against_persons(end, authors):
            return {
                "publication_year_start": start,
                "publication_year_end": end,
                "publication_year_source": "summary_regex",
                "publication_year_confidence": "high",
            }

    year, source, confidence = _extract_single_year(text, SINGLE_YEAR_PATTERNS, authors)
    return {
        "publication_year_start": year,
        "publication_year_end": None,
        "publication_year_source": source,
        "publication_year_confidence": confidence,
    }


def _extract_composition_dates(summaries: List[str], authors: List[dict]) -> dict:
    """Extract original composition dates for translated works."""
    null_result = {
        "composition_year_start": None,
        "composition_year_end": None,
        "composition_year_source": None,
        "composition_year_confidence": None,
    }

    if not authors:
        return null_result

    text = " ".join(summaries)

    date_range = _extract_year_range(text)
    if date_range:
        start, end = date_range
        if _sanity_check_against_persons(
            start, authors
        ) and _sanity_check_against_persons(end, authors):
            return {
                "composition_year_start": start,
                "composition_year_end": end,
                "composition_year_source": "summary_regex",
                "composition_year_confidence": "low",
            }

    year, source, _ = _extract_single_year(text, SINGLE_YEAR_PATTERNS, authors)
    return {
        "composition_year_start": year,
        "composition_year_end": None,
        "composition_year_source": source,
        "composition_year_confidence": "low" if year is not None else None,
    }


def _extract_translation_year(
    summaries: List[str],
    translators: List[dict],
) -> Tuple[Optional[int], Optional[str]]:
    """Extract the translation year using translator dates for sanity checks."""
    text = " ".join(summaries)
    year, source, _ = _extract_single_year(text, TRANSLATION_PATTERNS, translators)
    return year, source


# ---------------------------------------------------------------------------
# Metadata – public API
# ---------------------------------------------------------------------------


def _empty_metadata(ebook: str, error: str) -> dict:
    """Return a metadata dict with all fields set to None and an error message."""
    return {
        "id": None,
        "title": None,
        "authors": None,
        "editors": None,
        "translators": None,
        "is_translation": None,
        "languages": None,
        "subjects": None,
        "bookshelves": None,
        "copyright": None,
        "publication_year_start": None,
        "publication_year_end": None,
        "publication_year_source": None,
        "publication_year_confidence": None,
        "composition_year_start": None,
        "composition_year_end": None,
        "composition_year_source": None,
        "composition_year_confidence": None,
        "translation_year": None,
        "translation_year_source": None,
        "summary": None,
        "summary_is_unverified": None,
        "plain_text_url": None,
        "source_url": None,
        "error": error,
    }


def fetch_metadata(ebook_id: str) -> dict:
    """Fetch metadata from the Gutendex API for a given Gutenberg ebook ID.

    Returns a sidecar-ready metadata dict.
    An 'error' key is present only if something went wrong.
    """
    gutendex_url = f"https://gutendex.com/books/{ebook_id}"
    try:
        resp = requests.get(gutendex_url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return _empty_metadata(ebook_id, str(e))

    # -- Basic fields
    authors = data.get("authors", [])
    editors = data.get("editors", [])
    translators = data.get("translators", [])
    summaries = data.get("summaries", [])
    is_translation = len(translators) > 0

    # -- Plain text URL (UTF-8 preferred)
    formats = data.get("formats", {})
    plain_text_url = formats.get("text/plain; charset=utf-8")

    # -- Source URL
    source_url = f"https://www.gutenberg.org/ebooks/{data.get('id', ebook_id)}"

    # -- Date extraction
    summary_text = summaries[0] if summaries else ""

    if is_translation:
        pub_dates = {
            "publication_year_start": None,
            "publication_year_end": None,
            "publication_year_source": None,
            "publication_year_confidence": None,
        }
        comp_dates = _extract_composition_dates(summaries, authors)
        translation_year, translation_year_source = _extract_translation_year(
            summaries, translators
        )
    else:
        pub_dates = _extract_publication_dates(summaries, authors)
        comp_dates = {
            "composition_year_start": None,
            "composition_year_end": None,
            "composition_year_source": None,
            "composition_year_confidence": None,
        }
        translation_year, translation_year_source = None, None

    return {
        "id": data.get("id"),
        "title": data.get("title"),
        "authors": authors,
        "editors": editors,
        "translators": translators,
        "is_translation": is_translation,
        "languages": data.get("languages", []),
        "subjects": data.get("subjects", []),
        "bookshelves": data.get("bookshelves", []),
        "copyright": data.get("copyright"),
        **pub_dates,
        **comp_dates,
        "translation_year": translation_year,
        "translation_year_source": translation_year_source,
        "summary": summary_text or None,
        "summary_is_unverified": True if summary_text else None,
        "plain_text_url": plain_text_url,
        "source_url": source_url,
    }


def save_metadata(
    metadata: Dict[str, Optional[str]],
    output_dir: str = "metadata",
) -> str:
    """Write *metadata* to ``<output_dir>/<ebook_id>.metadata.json``.

    Returns the path to the written file.
    """
    ebook_id = metadata.get("id")
    if ebook_id is not None:
        ebook_id = str(ebook_id)
    else:
        ebook_id = _extract_ebook_id(metadata.get("source_url", ""))
    if not ebook_id:
        raise ValueError("Cannot extract ebook ID from metadata dict")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_file = output_path / f"{ebook_id}.metadata.json"
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return str(metadata_file)


def plaintext_url(ebook_id: str) -> str:
    """Return the canonical Gutenberg CDN plain-text URL for an ebook ID.

    Example: '11' -> 'https://www.gutenberg.org/cache/epub/11/pg11.txt'
    """
    return f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"


# ---------------------------------------------------------------------------
# Preliminary text cleaning with regex and heuristics
# ---------------------------------------------------------------------------


def strip_boilerplate(text: str) -> str:
    """Remove Project Gutenberg's header and footer boilerplate.

    Slices between the standard *** START/END *** markers.
    Returns the text stripped but otherwise unmodified if no markers are found.
    """
    start_marker = re.search(
        r"\*{3}\s*START OF (?:THE |THIS )?PROJECT GUTENBERG[^\n]*\*{3}",
        text,
        re.IGNORECASE,
    )
    end_marker = re.search(
        r"\*{3}\s*END OF (?:THE |THIS )?PROJECT GUTENBERG[^\n]*\*{3}",
        text,
        re.IGNORECASE,
    )

    if start_marker and end_marker and end_marker.start() > start_marker.end():
        return text[start_marker.end() : end_marker.start()].strip()

    return text.strip()


# ---------------------------------------------------------------------------
# Text normalisation & anchor finding
# ---------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Normalise whitespace for reliable string matching.

    - Carriage returns become newlines.
    - Runs of three or more consecutive newlines are collapsed to two.
    - All remaining non-newline whitespace is collapsed to a single space.
    """
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


def _map_norm_to_orig(original: str, normalised: str) -> dict:
    """Map non-whitespace character positions in *normalised* back to *original*.

    The two strings must differ only in whitespace. Returns ``{normalised_index: original_index}``.
    """
    char_map = {}
    orig_idx = 0
    for norm_idx, ch in enumerate(normalised):
        if ch.isspace():
            continue
        while original[orig_idx].isspace():
            orig_idx += 1
        char_map[norm_idx] = orig_idx
        orig_idx += 1
    return char_map


def _pick_best_occurrence(positions: List[int], text: str, anchor_len: int) -> int:
    """Return the first position where the anchor is followed by a blank line (``\\n\\n``).

    This discriminates real narrative headings from Table-of-Contents entries:
    a ToC entry is immediately followed by more text on the same line, whereas
    a genuine heading is separated from the prose below by a blank line.

    Falls back to ``positions[0]`` if no position qualifies.

    *text* must already be normalised; *anchor_len* is ``len(normalised_anchor)``.
    """
    for idx in positions:
        end = idx + anchor_len
        if text[end : end + 2] == "\n\n":
            return idx
    return positions[0]


def locate_anchor(text: str, marker: str, is_start: bool = True) -> Tuple[int, int]:
    """Locate *marker* inside *text* and return ``(position, length)``.

    Uses two strategies, both operating on normalised text to avoid ``\\r\\n``
    discrepancies. Positions are mapped back to the original via ``_map_norm_to_orig``.

    - Strategy 1 — full normalised match: search for all occurrences of the
      full marker. Start anchors use ``_pick_best_occurrence`` to skip ToC
      entries; end anchors use the last occurrence.
    - Strategy 2 — single-line fallback: if the full marker is not found,
      reduce it to its first (start) or last (end) non-empty line and retry.

    Returns ``(-1, 0)`` if the marker cannot be located.
    """
    norm_text = normalise(text)
    norm_marker = normalise(marker)
    char_map = _map_norm_to_orig(text, norm_text)

    # -- Strategy 1: full normalised match
    if is_start:
        positions = [m.start() for m in re.finditer(re.escape(norm_marker), norm_text)]
        if positions:
            best_idx = _pick_best_occurrence(positions, norm_text, len(norm_marker))
            if best_idx in char_map:
                return char_map[best_idx], len(marker)
    else:
        idx = norm_text.rfind(norm_marker)
        if idx != -1 and idx in char_map:
            return char_map[idx], len(marker)

    # -- Strategy 2: single-line fallback
    lines = [line for line in marker.split("\n") if line.strip()]
    if not lines:
        return -1, 0

    line_marker = normalise(lines[0] if is_start else lines[-1])

    positions = [m.start() for m in re.finditer(re.escape(line_marker), norm_text)]
    if not positions:
        return -1, 0

    if is_start:
        best_idx = _pick_best_occurrence(positions, norm_text, len(line_marker))
    else:
        best_idx = positions[-1]

    if best_idx not in char_map:
        return -1, 0

    return char_map[best_idx], len(line_marker)
