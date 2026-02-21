#!/usr/bin/env python3
"""Summarize book sections into structured JSON + Markdown notes using an LLM."""

import json
import os
import re
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from .project import OUTPUTS_ROOT, PROMPTS_ROOT, list_output_book_dirs

PROMPT_NAME = "summarize"  # prompts/summarize.md

TIMEOUT_SECONDS = 300  # per LLM call timeout
MAX_RETRIES = 3  # number of retry attempts on timeout/error
MAX_WORKERS = 8  # max concurrent threads per chapter


# ---------------------------------------------------------------------------
# Config / prompt loading
# ---------------------------------------------------------------------------


def load_prompt(name: str) -> str:
    path = PROMPTS_ROOT / f"{name}.md"
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
        temperature=0,
        timeout=TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content.strip()


def parse_llm_json(raw: str) -> dict:
    """Extract JSON from LLM response, handling optional code fences."""
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw, count=1)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, count=1)
    return json.loads(cleaned)


def process_section(
    client: OpenAI,
    model: str,
    system_prompt: str,
    chapter_name: str,
    section: dict,
) -> dict:
    """Call the LLM for a single section with retry on timeout/error.
    Returns a parsed JSON dict. Raises when all retries are exhausted.
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
            except json.JSONDecodeError:
                # Non-retryable: LLM replied but returned bad JSON — store raw
                return {
                    "name": sec_name,
                    "summary": "",
                    "subsections": [{"name": sec_name, "content": raw}],
                    "code": None,
                    "key_terms": [],
                    "interview": None,
                    "more": [],
                    "updates": [],
                    "retained": [],
                    "omitted": [],
                    "_raw_response": True,
                }
        except Exception as e:
            last_err = e
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Section '{sec_name}' failed after {MAX_RETRIES} attempts: {e}"
                ) from e

    raise RuntimeError(f"Section '{sec_name}' exhausted retries")  # unreachable


# ---------------------------------------------------------------------------
# JSON → Markdown conversion
# ---------------------------------------------------------------------------


def section_json_to_markdown(data: dict) -> str:
    """Convert a single section's JSON output to Markdown."""
    lines: list[str] = []

    name = data.get("name", "Untitled")
    summary = data.get("summary", "")
    lines.append(f"# {name}")
    lines.append("")
    if summary:
        lines.append(summary)
        lines.append("")

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
                path = f"figure_{int(fig_id):04d}.png"
                lines.append(f"![{fig_caption}]({path})")
                lines.append("")

    code = data.get("code")
    if code and code.get("content"):
        lang = code.get("lang", "")
        lines.append(f"```{lang}")
        lines.append(code["content"])
        lines.append("```")
        lines.append("")

    key_terms = data.get("key_terms") or data.get("key_term")
    if key_terms:
        lines.append("__*Key Terms:*__")
        lines.append("")
        for item in key_terms:
            term = item.get("term", "")
            definition = item.get("definition", "")
            if term:
                lines.append(f"- **{term}**: {definition}")
            elif definition:
                lines.append(f"- {definition}")
            lines.append("")

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

    updates = data.get("updates") or data.get("update")
    if updates:
        lines.append("__*Update:*__")
        lines.append("")
        for item in updates:
            item_name = item.get("name", "")
            item_content = item.get("content", "")
            if item_name:
                lines.append(f"### {item_name}")
                lines.append("")
            if item_content:
                lines.append(item_content)
                lines.append("")

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
    parts = [section_json_to_markdown(s) for s in sections]
    return "\n\n---\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


def sanitize_filename(name: str) -> str:
    """Make a string safe for use as a filename."""
    name = re.sub(r"[<>:\"/\\|?*]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
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
    safe_name = sanitize_filename(f"{chapter_idx}. {chapter_name}")

    json_path = output_dir / f"{safe_name}.json"
    md_path = output_dir / f"{safe_name}.md"

    if json_path.exists():
        print(f"  Skipping (already exists): {safe_name}")
        return

    n = len(sections)
    slots: list[dict | None] = [None] * n

    pbar = tqdm(total=n, desc=f"  Ch {chapter_idx}", unit="sec", leave=True)
    future_deadline = TIMEOUT_SECONDS * MAX_RETRIES + 10

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, n)) as executor:
        future_to_idx: dict[Future, int] = {
            executor.submit(
                process_section, client, model, system_prompt, chapter_name, sec
            ): i
            for i, sec in enumerate(sections)
        }

        for future in as_completed(future_to_idx, timeout=future_deadline * n):
            idx = future_to_idx[future]
            sec_name = sections[idx]["name"]
            try:
                slots[idx] = future.result(timeout=future_deadline)
            except Exception as e:
                tqdm.write(f"  ERROR section '{sec_name}': {e}")
                slots[idx] = {
                    "name": sec_name,
                    "summary": f"ERROR: {e}",
                    "subsections": [],
                    "code": None,
                    "key_terms": [],
                    "interview": None,
                    "more": [],
                    "updates": [],
                    "retained": [],
                    "omitted": [],
                    "_error": str(e),
                }
            finally:
                pbar.update(1)

    pbar.close()

    section_results: list[dict] = [
        (
            slots[i]
            if slots[i] is not None
            else {
                "name": sec["name"],
                "summary": "MISSING",
                "subsections": [],
                "code": None,
                "key_terms": [],
                "interview": None,
                "more": [],
                "updates": [],
                "retained": [],
                "omitted": [],
            }
        )
        for i, sec in enumerate(sections)
    ]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"chapter": chapter_name, "model": model, "sections": section_results},
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(chapter_to_markdown(chapter_name, chapter_idx, section_results))

    print(f"  Wrote: {json_path.name}  |  {md_path.name}")


def _select_books(book):
    if not OUTPUTS_ROOT.exists():
        print(f"Error: outputs directory not found: {OUTPUTS_ROOT}")
        return [], []
    candidates = list_output_book_dirs(book)

    runnable = []
    skipped = []
    for book_dir in candidates:
        content_path = book_dir / "content.json"
        if content_path.exists():
            runnable.append(book_dir)
        else:
            skipped.append(book_dir.name)
    return runnable, skipped


def _select_chapters(chapters, chapter_indexes):
    if chapter_indexes:
        selected = []
        for idx in chapter_indexes:
            if idx < 1 or idx > len(chapters):
                print(
                    f"Warning: chapter {idx} out of range (1-{len(chapters)}), skipping"
                )
                continue
            selected.append((idx, chapters[idx - 1]))
        return selected
    return [(i + 1, ch) for i, ch in enumerate(chapters)]


def format_toc(chapters: list[dict]) -> str:
    """Render a compact TOC string for prompt injection."""
    lines: list[str] = []
    for i, chapter in enumerate(chapters, 1):
        lines.append(f"{i}. {chapter.get('name', 'Untitled Chapter')}")
        for j, section in enumerate(chapter.get("sections") or [], 1):
            lines.append(f"  {i}.{j} {section.get('name', 'Untitled Section')}")
    return "\n".join(lines)


def run(book=None, chapter_indexes=None, prompt=PROMPT_NAME):
    load_dotenv()

    runnable, skipped = _select_books(book)

    if not runnable:
        print("Error: No books with content.json found under outputs/")
        return 1

    if not book:
        print(
            f"Found {len(runnable)} runnable books under outputs/ "
            f"(skipped {len(skipped)} without content.json)."
        )
        for name in skipped:
            print(f"  - Skipping {name}: missing outputs/{name}/content.json")

    client = build_client()
    model = os.environ.get("LLM_ID", "gpt-4o")
    failed_books = []

    for book_dir in runnable:
        content_path = book_dir / "content.json"
        with open(content_path, "r", encoding="utf-8") as f:
            book_data = json.load(f)

        chapters = book_data["chapters"]
        selected = _select_chapters(chapters, chapter_indexes)
        if not selected:
            print(f"\nBook: {book_dir.name}  ({len(chapters)} chapters)")
            print("No chapters to process.")
            continue

        system_prompt = (
            load_prompt(prompt)
            .replace("{{BOOK_NAME}}", book_dir.name)
            .replace("{{TOC}}", format_toc(chapters))
        )
        output_dir = book_dir / "chapters"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nBook: {book_dir.name}  ({len(chapters)} chapters)")
        print(f"Model: {model}")
        print(f"Output: {output_dir}")
        print(f"Chapters to process: {[idx for idx, _ in selected]}")
        print()

        try:
            for chapter_idx, chapter in selected:
                ch_name = chapter["name"]
                n_sections = len(chapter["sections"])
                print(f"Chapter {chapter_idx}: {ch_name} ({n_sections} sections)")
                process_chapter(
                    client, model, system_prompt, chapter, chapter_idx, output_dir
                )
                print()
        except Exception as exc:
            print(f"Error: failed processing book '{book_dir.name}': {exc}")
            failed_books.append(book_dir.name)

    print("Done.")
    if failed_books:
        print("Failed books:")
        for name in failed_books:
            print(f"  - {name}")
        return 1
    return 0
