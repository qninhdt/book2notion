#!/usr/bin/env python3
"""Convert book to JSON format using:
- TOC from books/{name}.pdf
- OCR data from outputs/{name}/hybrid_auto/{name}_content_list_v2.json

Uses pymupdf to extract TOC with page numbers, then matches TOC headings
to OCR title elements using soft matching constrained by expected page numbers.
"""
import json
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path

import fitz  # pymupdf

from .project import BOOKS_ROOT, OUTPUTS_ROOT, list_output_book_dirs

EXCLUDE_PATTERNS = [
    r"key\s+terms",
    r"review\s+questions",
    r"problems$",
    r"key\s+terms.*?review\s+questions.*?problems",
    r"key\s+terms\s+and\s+problems",
    r"key\s+terms\s+and\s+review",
    r"learning\s+objectives",
    r"recommended\s+reading",
    r"^references$",
    r"^summary$",
    r"^conclusion$",
    r"^homework",
    r"^mlfq:\s*summary$",
]

TOC_BOOKEND_EXCLUDE_PATTERNS = [
    r"^cover$",
    r"^copyright$",
    r"^brief\s+contents$",
    r"^table\s+of\s+contents$",
    r"^contents$",
    r"^preface$",
    r"^acknowledg(e)?ments$",
    r"^references$",
    r"^bibliography$",
    r"^index$",
    r"^about\s+the\s+authors$",
    r"^colophon$",
]


def should_exclude_section(section_name):
    name_lower = section_name.lower().strip()
    return any(re.search(p, name_lower) for p in EXCLUDE_PATTERNS)


def _clean_toc_title(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _is_toc_bookend_title(title):
    t = _clean_toc_title(title).lower()
    return any(re.search(p, t) for p in TOC_BOOKEND_EXCLUDE_PATTERNS)


# ---------------------------------------------------------------------------
# TOC extraction from PDF using pymupdf (with page numbers)
# ---------------------------------------------------------------------------


def extract_toc_from_pdf(pdf_path):
    """Extract TOC from PDF using pymupdf. Returns chapters with sections,
    each having an expected page number (0-indexed, matching OCR page indices).
    pymupdf's get_toc returns 1-indexed pages, so we subtract 1.
    """
    doc = fitz.open(str(pdf_path))
    toc = doc.get_toc(simple=True)  # [level, title, page_1indexed]
    doc.close()

    if not toc:
        print("Warning: No TOC found in PDF")
        return {"chapters": []}

    normalized_toc = [
        (level, _clean_toc_title(title), page_1indexed)
        for level, title, page_1indexed in toc
    ]

    # Detect chapter candidates.
    chapter_indices = [
        i
        for i, (_, title, _) in enumerate(normalized_toc)
        if re.match(r"^Chapter\s+\d+[.:]?\s+", title)
    ]

    if not chapter_indices:
        chapter_indices = [
            i
            for i, (_, title, _) in enumerate(normalized_toc)
            if re.match(r"^\d+\.\s*\S", title) and not re.match(r"^\d+\.\d+", title)
        ]

    if not chapter_indices:
        chapter_indices = [
            i
            for i, (_, title, _) in enumerate(normalized_toc)
            if re.match(r"^\d+\s+\S", title)
        ]

    if not chapter_indices:
        # Fallback: use top-level TOC entries minus obvious front/back matter.
        min_level = min(level for level, _, _ in normalized_toc)
        chapter_indices = [
            i
            for i, (level, title, _) in enumerate(normalized_toc)
            if level == min_level and not _is_toc_bookend_title(title)
        ]

    if not chapter_indices:
        print("Warning: No chapters found in TOC")
        return {"chapters": []}

    chapters = []
    for ci_pos, ci in enumerate(chapter_indices):
        level, title, page_1indexed = normalized_toc[ci]
        page_0indexed = page_1indexed - 1

        ch_match = re.match(r"^Chapter\s+\d+[.:]?\s+(.+)$", title)
        if ch_match:
            chapter_name = ch_match.group(1).strip()
        else:
            ch_match2 = re.match(r"^(\d+)\.\s*(.+)$", title)
            if ch_match2:
                chapter_name = ch_match2.group(2).strip()
            else:
                ch_match3 = re.match(r"^(\d+)\s+(.+)$", title)
                chapter_name = (
                    ch_match3.group(2).strip() if ch_match3 else title.strip()
                )

        # Collect sections between this chapter and the next TOC peer.
        # Section depth is relative to each chapter entry (handles mixed TOC levels).
        section_level = level + 1
        next_ci = len(toc)
        for j in range(ci + 1, len(toc)):
            next_level = normalized_toc[j][0]
            if next_level <= level:
                next_ci = j
                break

        sections = []
        for j in range(ci + 1, next_ci):
            s_level, s_title, s_page = normalized_toc[j]
            if s_level == section_level:
                sections.append(
                    {
                        "name": s_title.strip(),
                        "page": s_page - 1,
                    }
                )

        if not sections:
            # Fallback for chapter-only TOCs.
            sections.append(
                {
                    "name": chapter_name,
                    "page": page_0indexed,
                }
            )

        chapters.append(
            {
                "name": chapter_name,
                "page": page_0indexed,
                "end_page": normalized_toc[next_ci][2] - 1
                if next_ci < len(normalized_toc)
                else None,
                "sections": sections,
            }
        )

    return {"chapters": chapters}


# ---------------------------------------------------------------------------
# OCR JSON loading
# ---------------------------------------------------------------------------

SKIP_TYPES = {"page_header", "page_footer", "page_number"}


def _get_text_from_inline(items):
    parts = []
    for item in items:
        if item.get("type") == "text":
            parts.append(item["content"])
        elif item.get("type") == "equation_inline":
            parts.append(item["content"])
    return "".join(parts).strip()


def load_ocr_elements(ocr_path):
    """Load OCR JSON. Returns (pages, flat_titles).
    flat_titles: list of {text, page_idx, elem_idx, flat_idx}
    """
    with open(ocr_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    flat_titles = []
    for page_idx, page in enumerate(pages):
        for elem_idx, elem in enumerate(page):
            if elem["type"] == "title":
                text = _get_text_from_inline(elem["content"].get("title_content", []))
                if text:
                    flat_titles.append(
                        {
                            "text": text,
                            "page_idx": page_idx,
                            "elem_idx": elem_idx,
                            "flat_idx": len(flat_titles),
                        }
                    )

    return pages, flat_titles


def _rewrite_and_copy_images_in_chapters(chapters, ocr_path):
    """Copy only images used in extracted content and rewrite to figure_XXXX names."""
    ocr_path = Path(ocr_path)
    output_book_dir = ocr_path.parent.parent
    target_dir = output_book_dir / "images"
    target_dir.mkdir(parents=True, exist_ok=True)

    path_map = {}  # original content image path -> rewritten image path
    copied = 0
    missing = 0

    image_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    def _replace_one(match):
        nonlocal copied, missing
        alt = match.group(1)
        src_rel = match.group(2).strip()
        if (
            not src_rel
            or src_rel.startswith("http://")
            or src_rel.startswith("https://")
        ):
            return match.group(0)

        if src_rel in path_map:
            return f"![{alt}]({path_map[src_rel]})"

        src_path = ocr_path.parent / src_rel
        if not src_path.exists() or not src_path.is_file():
            print(f"Warning: Invalid image file path: {src_path}")
            missing += 1
            return match.group(0)

        ext = src_path.suffix.lower() or ".png"
        new_name = f"figure_{len(path_map) + 1:04d}{ext}"
        new_rel = f"images/{new_name}"
        target_path = target_dir / new_name

        if not target_path.exists():
            shutil.copy2(src_path, target_path)
            copied += 1

        path_map[src_rel] = new_rel
        return f"![{alt}]({new_rel})"

    for chapter in chapters:
        for section in chapter.get("sections", []):
            content = section.get("content", "")
            if not content:
                continue
            section["content"] = image_re.sub(_replace_one, content)

    print(
        f"   Image normalize: {len(path_map)} used, {copied} copied, {missing} missing"
    )


# ---------------------------------------------------------------------------
# Soft matching with page-number constraints
# ---------------------------------------------------------------------------


def _normalize_for_match(text):
    text = text.lower().strip()
    text = re.sub(r"[\.\s]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[•\-]\s+", "", text)
    # Strip surrounding quotes (OCR sometimes wraps headings in quotes)
    text = text.strip('"').strip("\u201c").strip("\u201d").strip("'")
    return text


def _match_score(toc_title, ocr_title):
    a = _normalize_for_match(toc_title)
    b = _normalize_for_match(ocr_title)
    if a == b:
        return 1.0
    # Substring match only counts if the shorter string is reasonably long
    if len(a) >= 5 and len(b) >= 5 and (a in b or b in a):
        return 0.95
    return SequenceMatcher(None, a, b).ratio()


def _find_best_title_near_page(
    toc_name,
    flat_titles,
    expected_page,
    search_after_idx=-1,
    page_tolerance=5,
    threshold=0.70,
):
    """Find the best matching OCR title near the expected page.
    Search constraints:
      - Only consider titles on pages in [expected_page - 2, expected_page + page_tolerance]
      - Only consider titles after search_after_idx (sequential ordering)
      - Among candidates, prefer highest score; tie-break by last on closest page.
    Returns (flat_idx, score) or (-1, 0).
    """
    page_min = expected_page - 2
    page_max = expected_page + page_tolerance

    best_idx = -1
    best_score = 0
    best_page_dist = float("inf")

    for t in flat_titles:
        if t["flat_idx"] <= search_after_idx:
            continue
        if t["page_idx"] < page_min or t["page_idx"] > page_max:
            continue

        score = _match_score(toc_name, t["text"])
        if score < threshold:
            continue

        page_dist = abs(t["page_idx"] - expected_page)

        # Prefer: higher score → closer page → later element (last on page)
        if (
            (score > best_score)
            or (score == best_score and page_dist < best_page_dist)
            or (
                score == best_score
                and page_dist == best_page_dist
                and t["flat_idx"] > best_idx
            )
        ):
            best_idx = t["flat_idx"]
            best_score = score
            best_page_dist = page_dist

    return best_idx, best_score


# ---------------------------------------------------------------------------
# OCR element → Markdown conversion
# ---------------------------------------------------------------------------


def _inline_to_md(items):
    parts = []
    for item in items:
        t = item.get("type", "")
        if t == "text":
            parts.append(item.get("content", ""))
        elif t == "equation_inline":
            latex = item.get("content", "")
            parts.append(f" ${latex}$ ")
    return "".join(parts).strip()


def _elem_to_md(elem):
    etype = elem["type"]

    if etype in SKIP_TYPES:
        return ""

    if etype == "title":
        text = _get_text_from_inline(elem["content"].get("title_content", []))
        if text:
            return f"\n\n**{text}**\n\n"
        return ""

    content = elem.get("content", {})

    if etype == "paragraph":
        text = _inline_to_md(content.get("paragraph_content", []))
        return text + "\n\n" if text else ""

    if etype == "list":
        items = content.get("list_items", [])
        lines = []
        for item in items:
            item_text = _inline_to_md(item.get("item_content", []))
            item_text = re.sub(r"^[•\-\*]\s*", "", item_text)
            if item_text:
                lines.append(f"* {item_text}")
        return "\n".join(lines) + "\n\n" if lines else ""

    if etype == "image":
        src = content.get("image_source", {}).get("path", "")
        caption_items = content.get("image_caption", [])
        caption = _inline_to_md(caption_items) if caption_items else ""
        if src:
            return f"\n\n![{caption}]({src})\n\n"
        return ""

    if etype == "code":
        lang = content.get("code_language", "")
        code_text = _inline_to_md(content.get("code_content", []))
        if code_text:
            return f"\n```{lang}\n{code_text}\n```\n\n"
        return ""

    if etype == "algorithm":
        algo_text = _inline_to_md(content.get("algorithm_content", []))
        if algo_text:
            return f"\n```\n{algo_text}\n```\n\n"
        return ""

    if etype == "table":
        html = content.get("html", "")
        caption_items = content.get("table_caption", [])
        caption = _inline_to_md(caption_items) if caption_items else ""
        if html:
            result = ""
            if caption:
                result += f"**{caption}**\n\n"
            result += html + "\n\n"
            return result
        img_src = content.get("image_source", {}).get("path", "")
        if img_src:
            result = ""
            if caption:
                result += f"**{caption}**\n\n"
            result += f"![{caption}]({img_src})\n\n"
            return result
        return ""

    if etype == "page_footnote":
        text = _inline_to_md(content.get("page_footnote_content", []))
        return f"\n> {text}\n\n" if text else ""

    if etype == "equation_interline":
        latex = content.get("math_content", "")
        img_src = content.get("image_source", {}).get("path", "")
        if latex:
            return f"\n$$\n{latex}\n$$\n\n"
        if img_src:
            return f"\n![equation]({img_src})\n\n"
        return ""

    return ""


def _extract_content_range(pages, start_page, start_elem, end_page, end_elem):
    """Extract markdown content from OCR elements between two positions.
    start position is inclusive of (start_page, start_elem+1) — we skip the title itself.
    end position is exclusive.
    """
    parts = []

    if end_page is None:
        end_page = len(pages) - 1
        end_elem = len(pages[end_page]) if end_page < len(pages) else 0

    for page_idx in range(start_page, end_page + 1):
        if page_idx >= len(pages):
            break
        page = pages[page_idx]

        e_start = 0
        e_end = len(page)

        if page_idx == start_page:
            e_start = start_elem + 1
        if page_idx == end_page:
            e_end = end_elem

        for elem_idx in range(e_start, e_end):
            if elem_idx >= len(page):
                break
            md = _elem_to_md(page[elem_idx])
            if md.strip():
                parts.append(md)

    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Main parsing: match TOC sections to OCR titles, extract content
# ---------------------------------------------------------------------------


def _strip_chapter_prefix(text):
    t = (text or "").strip()
    t = re.sub(r"^chapter\s+\d+[.:]?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\d+[.:]?\s+", "", t)
    return t.strip()


def _strip_section_prefix(text):
    t = (text or "").strip()
    t = re.sub(r"^\d+(?:\.\d+)+(?:[.:])?\s+", "", t)
    t = re.sub(r"^\d+[.:]?\s+", "", t)
    return t.strip()


def _normalize_titles_without_indices(chapters):
    for chapter in chapters:
        ch_name = chapter.get("name", "").strip()
        if ch_name:
            chapter["name"] = _strip_chapter_prefix(ch_name)

        for section in chapter.get("sections", []):
            sec_name = section.get("name", "").strip()
            if sec_name:
                section["name"] = _strip_section_prefix(sec_name)
    return chapters


def parse_ocr_with_toc(ocr_path, toc_result):
    toc_chapters = toc_result["chapters"]

    print("\n2. Loading OCR JSON...")
    pages, flat_titles = load_ocr_elements(ocr_path)
    print(f"   {len(pages)} pages, {len(flat_titles)} title elements")

    print("\n3. Matching TOC sections to OCR titles...")

    # Build a flat list of ALL sections (including excluded ones as boundary markers)
    all_sections = []
    for ch_i, chapter in enumerate(toc_chapters):
        for sec_i, section in enumerate(chapter["sections"]):
            all_sections.append(
                {
                    "ch_idx": ch_i,
                    "sec_idx": sec_i,
                    "name": section["name"],
                    "expected_page": section["page"],
                    "excluded": should_exclude_section(section["name"]),
                }
            )

    # Match each section sequentially (including excluded ones for boundary tracking)
    matched_sections = (
        []
    )  # {ch_idx, name, flat_title_idx, page_idx, elem_idx, excluded}
    prev_matched_flat_idx = -1

    for sec_info in all_sections:
        flat_idx, score = _find_best_title_near_page(
            sec_info["name"],
            flat_titles,
            sec_info["expected_page"],
            search_after_idx=prev_matched_flat_idx,
        )

        if flat_idx < 0:
            ch_name = toc_chapters[sec_info["ch_idx"]]["name"]
            tag = " (excluded)" if sec_info["excluded"] else ""
            print(
                f"    WARNING: '{sec_info['name']}'{tag} (expected page {sec_info['expected_page']}) "
                f"not found in chapter '{ch_name}'"
            )
            continue

        t = flat_titles[flat_idx]
        ch_name = toc_chapters[sec_info["ch_idx"]]["name"]
        tag = " [boundary]" if sec_info["excluded"] else ""
        print(
            f"    [{ch_name}] '{sec_info['name']}' → "
            f"'{t['text']}' (page {t['page_idx']}, score={score:.2f}){tag}"
        )

        matched_sections.append(
            {
                "ch_idx": sec_info["ch_idx"],
                "name": sec_info["name"],
                "flat_title_idx": flat_idx,
                "page_idx": t["page_idx"],
                "elem_idx": t["elem_idx"],
                "excluded": sec_info["excluded"],
            }
        )
        prev_matched_flat_idx = flat_idx

    content_sections = [ms for ms in matched_sections if not ms["excluded"]]
    print(f"\n4. Extracting content for {len(content_sections)} sections...")

    result_chapters = {}  # ch_idx → {name, sections:[]}

    for ms_pos, ms in enumerate(matched_sections):
        if ms["excluded"]:
            continue

        # End boundary: next matched section's position (including excluded ones)
        end_page = None
        end_elem = None
        if ms_pos + 1 < len(matched_sections):
            nxt = matched_sections[ms_pos + 1]
            end_page = nxt["page_idx"]
            end_elem = nxt["elem_idx"]
        else:
            chapter_end_page = toc_chapters[ms["ch_idx"]].get("end_page")
            if chapter_end_page is not None and chapter_end_page >= ms["page_idx"]:
                end_page = chapter_end_page
                end_elem = 0

        content = _extract_content_range(
            pages, ms["page_idx"], ms["elem_idx"], end_page, end_elem
        )

        ch_idx = ms["ch_idx"]
        ch_name = toc_chapters[ch_idx]["name"]

        if ch_idx not in result_chapters:
            result_chapters[ch_idx] = {"name": ch_name, "sections": []}

        if content:
            result_chapters[ch_idx]["sections"].append(
                {"name": ms["name"], "content": content}
            )
            print(f"     [{ch_name}] {ms['name']}: {len(content)} chars")
        else:
            print(f"     [{ch_name}] {ms['name']}: WARNING - empty content")

    # Return chapters in order, only those with sections
    ordered = [
        result_chapters[k]
        for k in sorted(result_chapters.keys())
        if result_chapters[k]["sections"]
    ]
    ordered = _normalize_titles_without_indices(ordered)
    _rewrite_and_copy_images_in_chapters(ordered, ocr_path)
    return ordered


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _normalize_book_name(book_input):
    raw = str(book_input).strip().rstrip("/\\")
    if not raw:
        return raw
    p = Path(raw)
    if p.suffix.lower() == ".pdf":
        return p.stem
    return p.name


def _resolve_input_paths(name):
    pdf_path = BOOKS_ROOT / f"{name}.pdf"
    ocr_path = OUTPUTS_ROOT / name / "hybrid_auto" / f"{name}_content_list_v2.json"
    return pdf_path, ocr_path


def convert_book(book_name):
    name = _normalize_book_name(book_name)
    if not name:
        print("Error: Empty book name")
        return False

    pdf_path, ocr_path = _resolve_input_paths(name)

    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        return False
    if not ocr_path.exists():
        print(f"Error: OCR JSON not found: {ocr_path}")
        return False

    print(f"\nProcessing book: {name}")
    print("=" * 60)

    print("\n1. Extracting TOC from PDF...")
    toc_result = extract_toc_from_pdf(pdf_path)
    toc_chapters = toc_result["chapters"]
    print(f"   Found {len(toc_chapters)} chapters")

    if not toc_chapters:
        print("Error: Could not extract TOC from PDF")
        return False

    for ch in toc_chapters:
        print(f"   - {ch['name']} ({len(ch['sections'])} sections)")

    chapters = parse_ocr_with_toc(ocr_path, toc_result)

    print("\n\n5. Creating JSON output...")
    output_data = {"chapters": chapters}

    output_path = OUTPUTS_ROOT / name / "content.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    total_sections = sum(len(ch["sections"]) for ch in chapters)
    total_chars = sum(len(s["content"]) for ch in chapters for s in ch["sections"])

    print(f"\n{'=' * 60}")
    print(f"Successfully created {output_path}")
    print(f"  Chapters: {len(chapters)}")
    print(f"  Sections: {total_sections}")
    print(f"  Total content: {total_chars:,} characters")
    return True


def _discover_books():
    if not OUTPUTS_ROOT.exists():
        print(f"Error: outputs directory not found: {OUTPUTS_ROOT}")
        return [], []

    book_dirs = list_output_book_dirs()
    runnable = []
    skipped = []
    for book_dir in book_dirs:
        pdf_path, ocr_path = _resolve_input_paths(book_dir.name)
        missing = []
        if not pdf_path.exists():
            missing.append(str(pdf_path))
        if not ocr_path.exists():
            missing.append(str(ocr_path))
        if missing:
            skipped.append((book_dir.name, missing))
        else:
            runnable.append(book_dir.name)
    return runnable, skipped


def run(book=None):
    if book:
        return 0 if convert_book(book) else 1

    runnable_books, skipped_books = _discover_books()
    if not runnable_books:
        print("Error: No books with complete inputs found.")
        return 1

    print(
        f"Found {len(runnable_books)} runnable books under outputs/ "
        f"(skipped {len(skipped_books)} with missing inputs)."
    )
    for book_name, missing in skipped_books:
        print(f"Skipping {book_name}:")
        for item in missing:
            print(f"  - missing {item}")

    failed_books = []
    for name in runnable_books:
        try:
            ok = convert_book(name)
        except Exception as exc:
            print(f"Error: Unexpected failure for {name}: {exc}")
            ok = False
        if not ok:
            failed_books.append(name)

    print(
        f"\nRun summary: success={len(runnable_books) - len(failed_books)}, "
        f"failed={len(failed_books)}, skipped={len(skipped_books)}"
    )
    if failed_books:
        print("Failed books:")
        for name in failed_books:
            print(f"  - {name}")
        return 1
    return 0
