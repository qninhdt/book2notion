#!/usr/bin/env python3
"""
Summarize book sections into structured JSON + Markdown notes using an LLM.

Usage:
    python summarize_book.py --book "DDIA"
    python summarize_book.py --book "DDIA" --chapter 1
    python summarize_book.py --book "DDIA" --chapter 1 --chapter 3

Environment variables:
    LLM_BASE_URL  – OpenAI-compatible API base URL
    LLM_API_KEY   – API key
    LLM_ID        – Model identifier (e.g. "gpt-4o", "claude-3-opus")
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPT_NAME = "summarize"  # prompts/summarize.md

TIMEOUT_SECONDS = 180  # per LLM call timeout
MAX_RETRIES = 3  # number of retry attempts on timeout/error
MAX_WORKERS = 8  # max concurrent threads per chapter


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        print(f"Error: prompt file not found: {path}")
        sys.exit(1)
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


def build_client() -> OpenAI:
    base_url = os.environ.get("LLM_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY")
    if not base_url or not api_key:
        print("Error: LLM_BASE_URL and LLM_API_KEY environment variables must be set")
        sys.exit(1)
    return OpenAI(base_url=base_url, api_key=api_key)


def call_llm(client: OpenAI, model: str, system_prompt: str, user_content: str) -> str:
    """Send a request to the LLM and return the raw response text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        timeout=TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content.strip()


def process_section(
    client: OpenAI,
    model: str,
    system_prompt: str,
    chapter_name: str,
    section: dict,
) -> dict:
    """
    Call the LLM for a single section with retry on timeout/error.
    Returns a parsed JSON dict. Raises on all retries exhausted.
    """
    sec_name = section["name"]
    sec_content = section["content"]
    user_message = (
        f"## Section: {sec_name}\n\n"
        f"Chapter: {chapter_name}\n\n"
        f"---\n\n"
        f"{sec_content}"
    )

    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = call_llm(client, model, system_prompt, user_message)
            try:
                parsed = parse_llm_json(raw)
                if not parsed.get("name"):
                    parsed["name"] = sec_name
                return parsed
            except json.JSONDecodeError as e:
                # Non-retryable: LLM replied but returned bad JSON — store raw
                return {
                    "name": sec_name,
                    "summary": "",
                    "subsections": [{"name": sec_name, "content": raw}],
                    "code": None,
                    "interview": None,
                    "more": [],
                    "retained": [],
                    "omitted": [],
                    "_raw_response": True,
                }
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                # Will retry
                pass
            else:
                raise RuntimeError(
                    f"Section '{sec_name}' failed after {MAX_RETRIES} attempts: {e}"
                ) from e

    raise RuntimeError(f"Section '{sec_name}' exhausted retries")  # unreachable


def parse_llm_json(raw: str) -> dict:
    """Extract JSON from LLM response, handling optional code fences."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw, count=1)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, count=1)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# JSON  →  Markdown conversion
# ---------------------------------------------------------------------------


def section_json_to_markdown(data: dict) -> str:
    """Convert a single section's JSON output to Markdown."""
    lines: list[str] = []

    # Heading + summary
    name = data.get("name", "Untitled")
    summary = data.get("summary", "")
    lines.append(f"# {name}")
    lines.append("")
    if summary:
        lines.append(summary)
        lines.append("")

    # Subsections
    for sub in data.get("subsections") or []:
        sub_name = sub.get("name", "")
        sub_content = sub.get("content", "")
        if sub_name:
            lines.append(f"## {sub_name}")
            lines.append("")
        if sub_content:
            lines.append(sub_content)
            lines.append("")
        for fig in sub.get("figures") or []:
            fig_id = fig.get("id")
            fig_caption = fig.get("caption", "")
            if fig_id is not None:
                path = f"images/image_{int(fig_id):04d}.jpeg"
                lines.append(f"![{fig_caption}]({path})")
                lines.append("")

    # Code block
    code = data.get("code")
    if code and code.get("content"):
        lang = code.get("lang", "")
        lines.append(f"```{lang}")
        lines.append(code["content"])
        lines.append("```")
        lines.append("")

    # Interview questions
    interview = data.get("interview")
    if interview:
        lines.append("__*Interview:*__")
        lines.append("")
        for qa in interview:
            q = qa.get("question", "")
            level = qa.get("level", "")
            a = qa.get("answer", "")
            level_tag = f" (level: {level})" if level else ""
            lines.append(f"> **Question:** {q}{level_tag}")
            lines.append(f"> **Answer:** {a}")
            lines.append("")

    # More
    more = data.get("more")
    if more:
        lines.append("__*More:*__")
        lines.append("")
        for item in more:
            item_name = item.get("name", "")
            item_content = item.get("content", "")
            if item_name:
                lines.append(f"### {item_name}")
                lines.append("")
            if item_content:
                lines.append(item_content)
                lines.append("")

    # Editorial logic
    lines.append("---")
    lines.append("")
    lines.append("Editorial Logic:")
    lines.append("")

    retained = data.get("retained") or []
    if retained:
        lines.append("Retained:")
        for r in retained:
            lines.append(f"- **{r.get('name', '')}**: {r.get('reason', '')}")
        lines.append("")

    omitted = data.get("omitted") or []
    if omitted:
        lines.append("Omitted:")
        for o in omitted:
            lines.append(f"- **{o.get('name', '')}**: {o.get('reason', '')}")
        lines.append("")

    return "\n".join(lines)


def chapter_to_markdown(
    chapter_name: str, chapter_idx: int, sections: list[dict]
) -> str:
    """Combine all section markdowns into one chapter markdown file."""
    parts: list[str] = []
    for section_data in sections:
        parts.append(section_json_to_markdown(section_data))
    return "\n\n---\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


def sanitize_filename(name: str) -> str:
    """Make a string safe for use as a filename."""
    name = re.sub(r"[<>:\"/\\|?*]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    # Truncate to 100 chars to avoid filesystem limits
    if len(name) > 100:
        name = name[:100].rstrip()
    return name


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_chapter(
    client: OpenAI,
    model: str,
    system_prompt: str,
    chapter: dict,
    chapter_idx: int,
    output_dir: Path,
):
    """Process one chapter: all sections in parallel threads, then write JSON + MD."""
    chapter_name = chapter["name"]
    sections = chapter["sections"]
    prefix = f"{chapter_idx}. "
    safe_name = sanitize_filename(f"{prefix}{chapter_name}")

    json_path = output_dir / f"{safe_name}.json"
    md_path = output_dir / f"{safe_name}.md"

    # Resume support: skip if already done
    if json_path.exists():
        print(f"  Skipping (already exists): {safe_name}")
        return

    n = len(sections)
    # slots[i] will hold the result for sections[i] once its future completes
    slots: list[dict | None] = [None] * n

    pbar = tqdm(total=n, desc=f"  Ch {chapter_idx}", unit="sec", leave=True)
    # Hard-cap per future: TIMEOUT_SECONDS per attempt × retries + small buffer
    future_deadline = TIMEOUT_SECONDS * MAX_RETRIES + 10

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, n)) as executor:
        # Map future → original section index so we can preserve TOC order
        future_to_idx: dict[Future, int] = {
            executor.submit(
                process_section,
                client,
                model,
                system_prompt,
                chapter_name,
                sec,
            ): i
            for i, sec in enumerate(sections)
        }

        for future in as_completed(future_to_idx, timeout=future_deadline * n):
            idx = future_to_idx[future]
            sec_name = sections[idx]["name"]
            try:
                result = future.result(timeout=future_deadline)
                slots[idx] = result
            except Exception as e:
                tqdm.write(f"  ERROR section '{sec_name}': {e}")
                slots[idx] = {
                    "name": sec_name,
                    "summary": f"ERROR: {e}",
                    "subsections": [],
                    "code": None,
                    "interview": None,
                    "more": [],
                    "retained": [],
                    "omitted": [],
                    "_error": str(e),
                }
            finally:
                pbar.update(1)

    pbar.close()

    # Fill any slots that somehow remained None (shouldn't happen, but be safe)
    section_results: list[dict] = []
    for i, sec in enumerate(sections):
        if slots[i] is not None:
            section_results.append(slots[i])
        else:
            section_results.append(
                {
                    "name": sec["name"],
                    "summary": "MISSING",
                    "subsections": [],
                    "code": None,
                    "interview": None,
                    "more": [],
                    "retained": [],
                    "omitted": [],
                }
            )

    # Write JSON
    chapter_output = {
        "chapter": chapter_name,
        "sections": section_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chapter_output, f, indent=2, ensure_ascii=False)

    # Write Markdown
    md_content = chapter_to_markdown(chapter_name, chapter_idx, section_results)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"  Wrote: {json_path.name}  |  {md_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Summarize book sections using LLM")
    parser.add_argument("--book", required=True, help="Book folder name under books/")
    parser.add_argument(
        "--chapter",
        type=int,
        action="append",
        default=None,
        help="Chapter index (1-based). Can be repeated. Default: all chapters.",
    )
    parser.add_argument(
        "--prompt",
        default=PROMPT_NAME,
        help=f"Prompt name (from prompts/ folder, without .md). Default: {PROMPT_NAME}",
    )
    args = parser.parse_args()

    # Load book data
    book_dir = Path(__file__).parent / "books" / args.book
    content_path = book_dir / "content.json"
    if not content_path.exists():
        print(f"Error: {content_path} not found. Run convert_book.py first.")
        sys.exit(1)

    with open(content_path, "r", encoding="utf-8") as f:
        book_data = json.load(f)

    chapters = book_data["chapters"]
    print(f"Book: {args.book}  ({len(chapters)} chapters)")

    # Determine which chapters to process
    if args.chapter:
        selected = []
        for idx in args.chapter:
            if idx < 1 or idx > len(chapters):
                print(
                    f"Warning: chapter {idx} out of range (1-{len(chapters)}), skipping"
                )
                continue
            selected.append((idx, chapters[idx - 1]))
    else:
        selected = [(i + 1, ch) for i, ch in enumerate(chapters)]

    if not selected:
        print("No chapters to process.")
        sys.exit(0)

    # Setup
    book_name = book_dir.name
    system_prompt = load_prompt(args.prompt).replace("{{BOOK_NAME}}", book_name)
    client = build_client()
    model = os.environ.get("LLM_ID", "gpt-4o")
    output_dir = book_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model}")
    print(f"Output: {output_dir}")
    print(f"Chapters to process: {[idx for idx, _ in selected]}")
    print()

    for chapter_idx, chapter in selected:
        ch_name = chapter["name"]
        n_sections = len(chapter["sections"])
        print(f"Chapter {chapter_idx}: {ch_name} ({n_sections} sections)")
        process_chapter(client, model, system_prompt, chapter, chapter_idx, output_dir)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
