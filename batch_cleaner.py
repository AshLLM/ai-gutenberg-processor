"""Batch-process multiple Project Gutenberg ebooks through the cleaning pipeline."""

import os
import time
from urllib import request
from helper import (
    setup_openai,
    fetch_metadata,
    save_metadata,
    plaintext_url,
    strip_boilerplate,
    locate_anchor,
)


def parse_ebook_ids(user_input: str) -> list[str]:
    """Parse a comma-separated list or a range (e.g. '1-10') into ebook IDs."""
    user_input = user_input.strip()
    if "-" in user_input and "," not in user_input:
        start, end = user_input.split("-", 1)
        return [str(i) for i in range(int(start), int(end) + 1)]
    return [id.strip() for id in user_input.split(",") if id.strip()]


def _save_text(
    ebook_id: str,
    text: str,
    output_dir: str,
    suffix: str,
) -> str:
    """Write a text artifact and return the saved file path."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{ebook_id}_{suffix}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return file_path


# ── Prompt templates ────────────────────────────────────────────────────────


def _start_mapper_prompt(head: str, title: str = "") -> str:
    title_context = ""
    if title:
        title_context = f'\n<context>\nThe work\'s title is: "{title}"\n</context>\n'
    return f"""
<task_objective>
Decompose the text into **Structural Segments**.
Map the transition from front matter to the literary narrative.
</task_objective>
{title_context}
<segment_classification_rules>
- **EDITORIAL**: content not originating from the author (metadata, forewords, copyright information, transcriber notes, contents pages, prolegomenon, frontispiece, publisher information).
- **AUTHORIAL**: content originating from the author, including the work's own title line, main narrative, chapters, poems, prologues, prefaces, epigraphs, introductions, half title.
- **AUTHORIAL (ISLAND)**: Very short authorial fragments immediately followed by EDITORIAL segments.
- **Contiguity Principle**: Any title, subtitle, or heading that immediately precedes body text with no intervening editorial material is the opening of that body text. It MUST be included in the same AUTHORIAL segment — never split off as its own segment or classified as EDITORIAL.
</segment_classification_rules>

<mapping_instructions>
1. Analyze the text.
2. List every distinct heading or section as a numbered segment.
3. If a "Sandwich" (Authorial -> Editorial -> Authorial) occurs, precisely delineate the breaks.
</mapping_instructions>

<output_format>
Return a numbered list only, for example:
1. [Segment Name] (EDITORIAL)
2. [Segment Name] (AUTHORIAL - ISLAND)
3. [Segment Name] (EDITORIAL)
4. [Segment Name] (AUTHORIAL)
</output_format>

TEXT:
{head[:50000]}
"""


def _start_selector_prompt(start_segments: str) -> str:
    return f"""
<task>
Select the "Narrative Anchor": the specific segment where the main literary work begins.
</task>

<input_analysis>
{start_segments}
</input_analysis>

<selection_logic>
1. **Continuity Test:** Locate the LAST (EDITORIAL) segment. The (AUTHORIAL) segment immediately following it is the primary candidate.
2. **Island Bypass:** If an (AUTHORIAL - ISLAND) appears early but is separated from the main body by more EDITORIAL text, skip it.
3. **The Mainland Rule:** Select the segment that initiates the final, unbroken sequence of (AUTHORIAL) content.
</selection_logic>

<priority_ranking>
- Priority 1: First Chapter/Book heading after the final editorial interruption.
- Priority 2: Authorial Preface leading directly into narrative segments.
- Priority 3: First Authorial segment (if no Editorial segments exist).
</priority_ranking>

<output_rule>
Return ONLY the exact name of the segment. Return 'NULL' if no narrative start is found.
</output_rule>
"""


def _start_extractor_prompt(start_segment: str, head: str) -> str:
    return f"""
<task>
Extract the exact literal starting string for the segment: '{start_segment}'.
</task>

<extraction_rules>
- **Zero-Tolerance for Hallucination**: Do not summarise or fix typos; precision and accuracy are key.
- **Starting Point**: If the work's own title or heading appears on the line(s) immediately before the segment's body text, include it. Begin from the earliest such title line.
- **Length**: Provide exactly the first 5 lines of the text from this segment.
- **Precision**: Maintain all original whitespace, capitalisation, and line breaks.
</extraction_rules>

<uniqueness_verification>
Ensure the extracted snippet is unique. If '{start_segment}' appears in a Table of Contents (EDITORIAL), ignore it and find the version that starts the actual (AUTHORIAL) text block.
</uniqueness_verification>

<output_restriction>
Return ONLY the raw text. No labels. No preamble.
</output_restriction>

TEXT:
{head[:15000]}
"""


def _end_mapper_prompt(tail: str) -> str:
    return f"""
<task_objective>
Decompose the text into **Structural Segments**.
Map the transition from the literary narrative to the back matter.
Identify every distinct section in the final 25,000 characters.
</task_objective>

<segment_classification_rules>
- **AUTHORIAL**: content originating from the author (title, main narrative, afterwords, epilogues, authorial postscripts, authorial appendices).
- **AUTHORIAL (ISLAND)**: Short authorial fragments (e.g., "The End") that are followed by EDITORIAL text.
- **EDITORIAL**: content not originating from the author (non-authorial appendices, glossaries, index, endnotes, acknowledgments, transcriber notes, metadata).
</segment_classification_rules>

<output_format>
Return a numbered list of the final segments:
1. [Segment Name] (AUTHORIAL)
2. [Segment Name] (AUTHORIAL - ISLAND)
3. [Segment Name] (EDITORIAL)
</output_format>

TEXT:
{tail[-25000:]}
"""


def _end_selector_prompt(end_segments: str) -> str:
    return f"""
<task>
Based on the segment analysis, identify the **Exit Anchor**: the specific segment where the narrative officially concludes.
</task>

<input_analysis>
{end_segments}
</input_analysis>

<selection_logic>
1. **The Exit Test:** Identify the LAST segment labeled (AUTHORIAL) or (AUTHORIAL - ISLAND) that occurs before the final block of (EDITORIAL) text begins.
2. **Postscript Rule:** If a segment is an "Epilogue" or "Author's Note," it should be included as the exit anchor unless it is separated from the story by a significant EDITORIAL block.
3. **The Finality Rule:** We want the very last bit of text that came from the author's pen.
</selection_logic>

<output_rule>
Return ONLY the exact name of the segment. No extra text.
</output_rule>
"""


def _end_extractor_prompt(end_segment: str, tail: str) -> str:
    return f"""
<task>
Extract the exact literal CLOSING string for the segment: '{end_segment}'.
</task>

<extraction_rules>
- **Zero-Tolerance for Hallucination**: Return only what is present in the TEXT.
- **Precision**: You are looking for the LAST 5 lines of the '{end_segment}' section.
- **Verbatim**: Maintain all original whitespace, capitalisation, and punctuation.
- **Constraint**: Do NOT include any text from the EDITORIAL segments that follow it.
</extraction_rules>

<output_restriction>
Return ONLY the raw text. No labels. No preamble.
</output_restriction>

TEXT:
{tail[-25000:]}
"""


# ── Pipeline ────────────────────────────────────────────────────────────────


def process_ebook(client, ebook_id: str) -> dict:
    """Run the full pipeline for one ebook. Returns a result dict."""

    # 1 · Metadata
    print("  [1/4] Fetching metadata...", end="", flush=True)
    meta = fetch_metadata(ebook_id)
    if meta.get("error"):
        return {"ebook_id": ebook_id, "status": "error", "error": meta["error"]}
    save_metadata(meta, output_dir="metadata")
    title = meta.get("title") or "Unknown"
    print(f" {title}")

    # 2 · Fetch & clean
    print("  [2/4] Fetching & cleaning text...", end="", flush=True)
    url = meta.get("plain_text_url") or plaintext_url(ebook_id)
    response = request.urlopen(url)
    raw = response.read().decode("utf-8-sig")
    raw_file_path = _save_text(ebook_id, raw, "raw_txts", "raw")
    cleaned_text = strip_boilerplate(raw)
    print(f" {len(cleaned_text):,} chars")

    head = cleaned_text[:50000]
    tail = cleaned_text[-50000:]

    # 3 · LLM boundary detection
    print("  [3/4] LLM boundary detection...", flush=True)

    primary_title = meta.get("title") or ""

    print("         Start mapper...", end="", flush=True)
    start_segments = client.responses.create(
        model="gpt-5-mini", input=_start_mapper_prompt(head, primary_title)
    ).output_text
    print(" done")

    print("         Start selector...", end="", flush=True)
    start_segment = client.responses.create(
        model="gpt-5-nano", input=_start_selector_prompt(start_segments)
    ).output_text.strip()
    print(f" → {start_segment}")

    print("         Start extractor...", end="", flush=True)
    start_anchor = client.responses.create(
        model="gpt-5-mini", input=_start_extractor_prompt(start_segment, head)
    ).output_text.strip()
    print(" done")

    print("         End mapper...", end="", flush=True)
    end_segments = client.responses.create(
        model="gpt-5-mini", input=_end_mapper_prompt(tail)
    ).output_text
    print(" done")

    print("         End selector...", end="", flush=True)
    end_segment = client.responses.create(
        model="gpt-5-nano", input=_end_selector_prompt(end_segments)
    ).output_text.strip()
    print(f" → {end_segment}")

    print("         End extractor...", end="", flush=True)
    end_anchor = client.responses.create(
        model="gpt-5-mini", input=_end_extractor_prompt(end_segment, tail)
    ).output_text.strip()
    print(" done")

    # 4 · Extract & save
    print("  [4/4] Extracting & saving...", end="", flush=True)
    start_idx, start_len = locate_anchor(cleaned_text, start_anchor, is_start=True)
    end_idx, end_len = locate_anchor(cleaned_text, end_anchor, is_start=False)

    if start_idx == -1 or end_idx == -1:
        print(" failed")
        return {
            "ebook_id": ebook_id,
            "status": "anchor_failed",
            "start_idx": start_idx,
            "end_idx": end_idx,
            "raw_file": raw_file_path,
        }

    text_core = cleaned_text[start_idx : end_idx + end_len]
    file_path = _save_text(ebook_id, text_core, "core_txts", "core")

    print(f" {len(text_core):,} chars saved")

    return {
        "ebook_id": ebook_id,
        "title": title,
        "status": "success",
        "characters": len(text_core),
        "file": file_path,
        "raw_file": raw_file_path,
    }


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    client = setup_openai()

    user_input = input("Enter ebook IDs (comma-separated or range like 1-10): ")
    ebook_ids = parse_ebook_ids(user_input)
    print(f"Processing {len(ebook_ids)} ebook(s): {ebook_ids}\n")

    results = []
    for ebook_id in ebook_ids:
        print(f"── {ebook_id} ", end="")
        try:
            t0 = time.time()
            result = process_ebook(client, ebook_id)
            elapsed = time.time() - t0
            result["seconds"] = round(elapsed, 1)
            results.append(result)
            if result["status"] == "success":
                print(
                    f"✓ {result['title']} ({result['characters']:,} chars) [{elapsed:.1f}s]"
                )
            else:
                print(
                    f"✗ {result['status']}: {result.get('error', 'anchor mismatch')} [{elapsed:.1f}s]"
                )
        except Exception as e:
            elapsed = time.time() - t0
            results.append(
                {
                    "ebook_id": ebook_id,
                    "status": "exception",
                    "error": str(e),
                    "seconds": round(elapsed, 1),
                }
            )
            print(f"✗ {e} [{elapsed:.1f}s]")

    # Summary
    total_time = sum(r.get("seconds", 0) for r in results)
    succeeded = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    print(
        f"\n── Done: {len(succeeded)} succeeded, {len(failed)} failed ({total_time:.1f}s total)"
    )
    for r in failed:
        print(f"   ✗ {r['ebook_id']}: {r.get('error', r['status'])}")
