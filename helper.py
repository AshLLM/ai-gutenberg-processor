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
from bs4 import BeautifulSoup
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


def _choose_genre(subjects: List[str]) -> str:
    """Return a single genre-like label from a list of subjects.

    Scans for common genre keywords; falls back to the first short subject.
    """
    if not subjects:
        return ""
    genre_keywords = [
        "fiction",
        "novel",
        "poetry",
        "drama",
        "science fiction",
        "fantasy",
        "mystery",
        "detective",
        "horror",
        "romance",
        "children",
        "biography",
        "historical",
        "adventure",
        "satire",
    ]
    lowered = [s.lower() for s in subjects]
    for kw in genre_keywords:
        for s in lowered:
            if kw in s:
                return subjects[lowered.index(s)]
    short_subjects = [s for s in subjects if len(s.split()) <= 4]
    return short_subjects[0] if short_subjects else subjects[0]


def _extract_ebook_id(url: str) -> Optional[str]:
    """Pull the numeric ebook ID from a Gutenberg URL.

    Works for both page URLs (ebooks/11) and file URLs (/epub/11/).
    """
    match = re.search(r"ebooks/(\d+)|/epub/(\d+)(?:/|$)", url)
    if match:
        return match.group(1) or match.group(2)
    return None


def _extract_publication_date(summary_text: str) -> Optional[str]:
    """Extract a 4-digit year from summary text such as 'first published in 1868'."""
    if not summary_text:
        return None
    patterns = [
        r"(?:first\s+)?published\s+in\s+(\d{4})",
        r"(?:originally\s+)?published\s+in\s+(\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, summary_text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Metadata – public API
# ---------------------------------------------------------------------------


def _empty_metadata(url: str, error: str) -> Dict[str, Optional[str]]:
    """Return a metadata dict with all fields set to None and an error message."""
    return {
        "title": None,
        "author": None,
        "language": None,
        "publication_date": None,
        "ebook_no": None,
        "subjects": None,
        "genre": None,
        "source_url": url,
        "error": error,
    }


def fetch_metadata(ebook_url: str) -> Dict[str, Optional[str]]:
    """Scrape a Gutenberg ebook page and return a metadata dict.

    Keys: title, author, language, publication_date, ebook_no,
          subjects (list), genre, source_url.
    An 'error' key is added if something goes wrong, rather than raising.
    """
    try:
        resp = requests.get(ebook_url, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return _empty_metadata(ebook_url, str(e))

    soup = BeautifulSoup(resp.text, "html.parser")

    title = author = language = publication_date = ebook_no = None
    subjects = []

    table = soup.find("table", id="about_book_table")
    if not table:
        return _empty_metadata(
            ebook_url,
            "Bibliographic table not found; page structure may be non-standard",
        )

    for tr in table.find_all("tr"):
        th = tr.find("th")
        td = tr.find("td")
        if not th or not td:
            continue
        label = _clean_text(th.get_text()).lower()
        value_text = _clean_text(td.get_text())

        if label == "author":
            author = re.sub(r",\s*\d{4}-\d{4}$", "", value_text).strip()
        elif label == "title":
            title = value_text
        elif label == "language":
            language = value_text
        elif label == "ebook-no.":
            ebook_no = value_text
        elif label == "subject":
            link = td.find("a", class_="block")
            if link:
                subjects.append(_clean_text(link.get_text()))

    # Deduplicate subjects, preserving insertion order
    subjects = list(dict.fromkeys(subjects))

    if not publication_date:
        summary_div = soup.find("div", class_="summary-text-container")
        if summary_div:
            publication_date = _extract_publication_date(summary_div.get_text())

    return {
        "title": title or None,
        "author": author or None,
        "language": language or None,
        "publication_date": publication_date or None,
        "ebook_no": ebook_no or None,
        "subjects": subjects or None,
        "genre": _choose_genre(subjects) or None,
        "source_url": ebook_url,
    }


def save_metadata(
    metadata: Dict[str, Optional[str]],
    output_dir: str = "metadata",
) -> str:
    """Write *metadata* to ``<output_dir>/<ebook_id>.metadata.json``.

    Returns the path to the written file.
    """
    ebook_id = _extract_ebook_id(metadata.get("source_url", ""))
    if not ebook_id:
        raise ValueError("Cannot extract ebook ID from source_url in metadata dict")
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
