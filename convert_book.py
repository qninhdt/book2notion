#!/usr/bin/env python3
"""
Convert book to JSON format using TOC from PDF and content from HTML
Directory structure: ./books/[book name]/{content.pdf, content.html, images/, content.json}

Supports two book formats:
  - numbered: "Chapter N Name" chapters + "N.N Section" sections (e.g., Computer Architecture)
  - named:    "Chapter N. Name" chapters + plain-text sections (e.g., DDIA/O'Reilly books)
"""
import json
import re
import sys
import base64
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString
import PyPDF2


# Sections to exclude (case-insensitive match on section name)
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
    r"^homework",
    r"^mlfq:\s*summary$",
]


def should_exclude_section(section_name):
    """Check if a section should be excluded based on its name"""
    name_lower = section_name.lower().strip()
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, name_lower):
            return True
    return False


# ---------------------------------------------------------------------------
# TOC extraction helpers
# ---------------------------------------------------------------------------


def _flatten_outline(outline, level=0):
    """Flatten PDF outline into a list of {title, level} dicts"""
    items = []
    for item in outline:
        if isinstance(item, list):
            items.extend(_flatten_outline(item, level + 1))
        elif isinstance(item, dict):
            title = item.get("/Title", "").strip()
            if title:
                items.append({"title": title, "level": level})
        else:
            try:
                title = item.title.strip() if hasattr(item, "title") else ""
                if title:
                    items.append({"title": title, "level": level})
            except Exception:
                pass
    return items


def _detect_book_format(all_items):
    """
    Detect book TOC format.
      'numbered' – chapters are "Chapter N Name", sections are "N.N Name"
      'named'    – chapters are "Chapter N. Name", sections are plain text
    """
    for item in all_items:
        title = item["title"]
        # Named format uses a period after chapter number: "Chapter 1. Title"
        if re.match(r"^Chapter\s+\d+\.\s+", title):
            return "named"
        # Numbered format: "Chapter 1 Title" (no period)
        if re.match(r"^Chapter\s+\d+\s+\S", title):
            return "numbered"
    # Check for "N. Title" format (e.g., OSTEP: "2. Introduction to Operating Systems")
    # Note: some entries may lack space after dot (e.g. "14.Interlude: Memory API")
    for item in all_items:
        title = item["title"]
        if re.match(r"^\d+\.\s*\S", title) and not re.match(r"^\d+\.\d+\s", title):
            return "numbered"
    return "named"  # default


def _extract_toc_numbered(all_items):
    """Extract chapters/sections for numbered-format books.
    Handles both 'Chapter N Title' (Computer Architecture) and 'N. Title' (OSTEP) formats.
    """
    chapters = []
    current_chapter = None

    for item in all_items:
        title = item["title"]

        # "Chapter N Title" format (Computer Architecture)
        if title.startswith("Chapter "):
            match = re.match(r"^Chapter\s+(\d+)\s+(.+)$", title)
            if match:
                chapter_num, chapter_name = match.groups()
                current_chapter = {
                    "number": chapter_num,
                    "name": chapter_name.strip(),
                    "sections": [],
                }
                chapters.append(current_chapter)

        # "N. Title" format (OSTEP) — not a section like "N.N"
        # Note: space after dot may be missing (e.g. "14.Interlude: Memory API")
        elif re.match(r"^\d+\.\s*\S", title) and not re.match(r"^\d+\.\d+", title):
            match = re.match(r"^(\d+)\.\s*(.+)$", title)
            if match:
                chapter_num, chapter_name = match.groups()
                current_chapter = {
                    "number": chapter_num,
                    "name": chapter_name.strip(),
                    "sections": [],
                }
                chapters.append(current_chapter)

        # Section: "N.N Title"
        elif re.match(r"^\d+\.\d+\s+", title) and current_chapter:
            match = re.match(r"^(\d+)\.(\d+)\s+(.+)$", title)
            if match:
                chapter_num, section_num, section_name = match.groups()
                if chapter_num == current_chapter["number"]:
                    current_chapter["sections"].append(
                        {
                            "number": f"{chapter_num}.{section_num}",
                            "name": section_name.strip(),
                        }
                    )

    return chapters


def _extract_toc_named(all_items):
    """
    Extract chapters/sections for named-format books (DDIA/O'Reilly style).
    Chapters: items matching "Chapter N. Title" (any nesting level).
    Sections: direct children of chapters in the outline hierarchy.
    """
    chapters = []
    current_chapter = None
    chapter_level = None

    for item in all_items:
        title = item["title"]
        level = item["level"]

        ch_match = re.match(r"^Chapter\s+\d+[.:]?\s+(.+)$", title)
        if ch_match:
            chapter_name = ch_match.group(1).strip()
            current_chapter = {"name": chapter_name, "sections": []}
            chapters.append(current_chapter)
            chapter_level = level
        elif current_chapter is not None and chapter_level is not None:
            if level == chapter_level + 1:
                # Direct child of chapter → top-level section
                current_chapter["sections"].append({"name": title.strip()})

    return chapters


def extract_toc_from_pdf(pdf_path):
    """
    Extract TOC from PDF outline/bookmarks.
    Returns {'format': 'numbered'|'named', 'chapters': [...]}
    """
    with open(pdf_path, "rb") as f:
        pdf = PyPDF2.PdfReader(f)

        if not pdf.outline:
            print("Warning: No TOC found in PDF")
            return {"format": "named", "chapters": []}

        all_items = _flatten_outline(pdf.outline)

    book_format = _detect_book_format(all_items)
    if book_format == "numbered":
        chapters = _extract_toc_numbered(all_items)
    else:
        chapters = _extract_toc_named(all_items)

    return {"format": book_format, "chapters": chapters}


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------


def save_base64_image(base64_str, output_dir, image_index):
    """Save base64 image to file and return the filename"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if base64_str.startswith("data:image/"):
        match = re.match(r"data:image/([^;]+);base64,(.+)", base64_str)
        if match:
            img_format, img_data = match.groups()
            img_bytes = base64.b64decode(img_data)
            filename = f"image_{image_index:04d}.{img_format}"
            filepath = output_dir / filename
            with open(filepath, "wb") as f:
                f.write(img_bytes)
            return filename
    return None


# ---------------------------------------------------------------------------
# HTML → Markdown conversion
# ---------------------------------------------------------------------------


def element_to_markdown(element, images_dir, image_counter):
    """Convert an HTML element to markdown text, handling images"""
    if isinstance(element, NavigableString):
        return str(element)

    if element.name == "img":
        src = element.get("src", "")
        alt = element.get("alt", "")
        if src.startswith("data:image/"):
            filename = save_base64_image(src, images_dir, image_counter[0])
            if filename:
                image_counter[0] += 1
                return f"\n\n![{alt}](images/{filename})\n\n"
        elif src:
            return f"\n\n![{alt}]({src})\n\n"
        return ""

    if element.name in ["strong", "b"]:
        inner = children_to_markdown(element, images_dir, image_counter)
        return f"**{inner.strip()}**" if inner.strip() else ""

    if element.name in ["em", "i"]:
        inner = children_to_markdown(element, images_dir, image_counter)
        return f"*{inner.strip()}*" if inner.strip() else ""

    if element.name == "br":
        return "\n"

    if element.name in ["ul", "ol"]:
        items = []
        for li in element.find_all("li", recursive=False):
            item_text = children_to_markdown(li, images_dir, image_counter).strip()
            if item_text:
                items.append(f"  * {item_text}")
        return "\n".join(items) + "\n\n" if items else ""

    if element.name == "table":
        rows = []
        for tr in element.find_all("tr"):
            cells = [td.get_text().strip() for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            return "\n" + "\n".join(rows) + "\n\n"
        return ""

    if element.name in [
        "p",
        "div",
        "span",
        "li",
        "td",
        "th",
        "blockquote",
        "figcaption",
        "figure",
        "section",
        "article",
    ]:
        inner = children_to_markdown(element, images_dir, image_counter)
        if element.name in ["p", "div", "blockquote"]:
            return inner.strip() + "\n\n" if inner.strip() else ""
        return inner

    # h3/h4/h5 sub-section headings within content → bold
    if element.name in ["h3", "h4", "h5"]:
        text = element.get_text().strip()
        # Skip page-number markers like "6 CHAPTER 1 / BASIC CONCEPTS..."
        if re.match(r"^\d+\s+CHAPTER\s+\d+", text):
            return ""
        if text:
            return f"\n\n**{text}**\n\n"
        return ""

    # h6 are typically page markers, skip
    if element.name == "h6":
        return ""

    # h2 sub-headings in content (ASIDE, TIP, CRUX boxes in OSTEP) → bold
    if element.name == "h2":
        text = element.get_text().strip()
        if text:
            return f"\n\n**{text}**\n\n"
        return ""

    # h1 shouldn't appear within section content; skip safely
    if element.name == "h1":
        return ""

    return children_to_markdown(element, images_dir, image_counter)


def children_to_markdown(element, images_dir, image_counter):
    """Convert all children of an element to markdown"""
    result = ""
    for child in element.children:
        result += element_to_markdown(child, images_dir, image_counter)
    return result


# ---------------------------------------------------------------------------
# Numbered-format parsing (Computer Architecture)
# ---------------------------------------------------------------------------


def build_section_index(soup):
    """
    Build a dict mapping section numbers (e.g. '1.1') to their heading tags.
    Looks for h2/h3 tags whose text starts with N.N pattern.
    """
    section_index = {}
    for tag in soup.find_all(["h2", "h3"]):
        text = re.sub(r"\s+", " ", tag.get_text().strip())
        match = re.match(r"^(\d+\.\d+)\s+", text)
        if match:
            section_num = match.group(1)
            section_index[section_num] = tag
    return section_index


def extract_section_content(heading_tag, images_dir, image_counter, stop_tag=None):
    """
    Extract content from after a numbered-format heading tag until the next
    section boundary.

    If stop_tag is provided (identity-based stopping), stops when that exact
    tag is encountered.  Otherwise falls back to heuristic boundary detection.
    """
    content_parts = []
    stop_id = id(stop_tag) if stop_tag is not None else None

    for sibling in heading_tag.find_next_siblings():
        # Identity-based stop (precise: used when we know the next section tag)
        if stop_id is not None and id(sibling) == stop_id:
            break

        if sibling.name == "h1":
            break

        if stop_id is not None:
            # When using explicit stop_tag, only stop at h1 or the tag itself.
            # h2/h3 that are ASIDE/TIP/CRUX boxes become content.
            # But skip page-marker h6 and render h2/h3/h4/h5 as bold.
            pass
        else:
            # Original heuristic for Computer Architecture (no stop_tag)
            if sibling.name in ["h2"]:
                break
            if sibling.name == "h3":
                sib_text = re.sub(r"\s+", " ", sibling.get_text().strip())
                if re.match(r"^\d+\.\d+\s+", sib_text):
                    break
                if sib_text.upper().startswith("CHAPTER "):
                    break
                if sib_text.upper().startswith("LEARNING OBJECTIVES"):
                    break
                # Non-section h3 (e.g. decorative) — skip without rendering
                continue

        # Skip h6 page markers (e.g. "6 CHAPTER 1 / BASIC CONCEPTS...")
        if sibling.name == "h6":
            text = sibling.get_text().strip()
            if re.match(r"^\d+\s+CHAPTER\s+\d+", text) or re.match(
                r"^[IVXLCDM]+\s+CONTENTS", text
            ):
                continue

        md = element_to_markdown(sibling, images_dir, image_counter)
        if md.strip():
            content_parts.append(md)

    return "\n".join(content_parts).strip()


def parse_html_numbered(html_path, toc_chapters, images_dir):
    """Parse HTML for numbered-format books (Computer Architecture, OSTEP)."""
    print("   Loading HTML file...")
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    print("   Building section index from HTML headings...")
    section_index = build_section_index(soup)
    print(f"   Found {len(section_index)} section headings in HTML")

    # Build ordered list of all section heading tags for stop-tag computation.
    # This ensures extraction stops precisely at the next section, even when
    # non-section h2/h3 (ASIDE, TIP, CRUX) appear in between (as in OSTEP).
    all_section_nums_sorted = sorted(
        section_index.keys(), key=lambda k: [int(x) for x in k.split(".")]
    )
    # Map each section number to the tag of the NEXT section in document order
    next_section_tag = {}
    for i, sn in enumerate(all_section_nums_sorted):
        if i + 1 < len(all_section_nums_sorted):
            next_section_tag[sn] = section_index[all_section_nums_sorted[i + 1]]
        else:
            next_section_tag[sn] = None

    image_counter = [1]
    result_chapters = []

    for chapter in toc_chapters:
        print(f"\n   Chapter {chapter['number']}: {chapter['name']}")
        result_sections = []

        for section in chapter["sections"]:
            sec_num = section["number"]
            sec_name = section["name"]

            if should_exclude_section(sec_name):
                print(f"     Skipping {sec_num} {sec_name} (excluded)")
                continue

            heading_tag = section_index.get(sec_num)
            if heading_tag:
                stop_tag = next_section_tag.get(sec_num)
                content = extract_section_content(
                    heading_tag, images_dir, image_counter, stop_tag=stop_tag
                )
                if content:
                    result_sections.append({"name": sec_name, "content": content})
                    print(f"     {sec_num} {sec_name}: {len(content)} chars")
                else:
                    print(f"     {sec_num} {sec_name}: WARNING - empty content")
                    result_sections.append({"name": sec_name, "content": ""})
            else:
                print(f"     {sec_num} {sec_name}: WARNING - heading not found in HTML")

        if result_sections:
            result_chapters.append(
                {"name": chapter["name"], "sections": result_sections}
            )

    return result_chapters


# ---------------------------------------------------------------------------
# Named-format parsing (DDIA / O'Reilly)
# ---------------------------------------------------------------------------


def _normalize_heading(text):
    """Normalize heading text for matching (lowercase, collapse whitespace)."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _find_heading(
    name, all_headings, start_idx=0, strip_chapter_prefix=False, max_level=None
):
    """
    Return (tag, index) of the first heading in all_headings[start_idx:]
    whose normalized text matches `name`.

    Args:
        strip_chapter_prefix: Also accept headings like "chapter 5 replication"
            when searching for "replication".
        max_level: If set (e.g., 3), only consider h1/h2/h3 (ignore h4/h5).
    """
    name_norm = _normalize_heading(name)
    for i in range(start_idx, len(all_headings)):
        tag = all_headings[i]
        if max_level is not None:
            level = int(tag.name[1]) if tag.name and tag.name[1:].isdigit() else 9
            if level > max_level:
                continue
        text = _normalize_heading(tag.get_text())
        if text == name_norm:
            return tag, i
        if strip_chapter_prefix:
            stripped = re.sub(r"^chapter\s+\d+[.:]?\s*", "", text).strip()
            if stripped == name_norm:
                return tag, i
    return None, -1


def _extract_until_tag(
    start_tag, stop_tag, chapter_heading_ids, images_dir, image_counter
):
    """
    Walk find_next_siblings() from start_tag, collecting markdown.
    Stops when:
      - we reach stop_tag (identity check)
      - we reach any known chapter heading
      - we reach h1/h2
    """
    content_parts = []
    stop_id = id(stop_tag) if stop_tag is not None else None

    for sibling in start_tag.find_next_siblings():
        if stop_id is not None and id(sibling) == stop_id:
            break
        if id(sibling) in chapter_heading_ids:
            break
        if sibling.name in ["h1", "h2"]:
            break

        md = element_to_markdown(sibling, images_dir, image_counter)
        if md.strip():
            content_parts.append(md)

    return "\n".join(content_parts).strip()


def _find_sections_for_chapter(toc_sections, all_headings, ch_pos, next_ch_pos):
    """
    Intelligently find section heading tags within a chapter's range.

    Problem: DDIA HTML has duplicate heading names at different levels —
    e.g., "Reliability" appears as a brief h4 intro inside "Thinking About
    Data Systems" AND again as the actual content h4 section.

    Two-phase strategy:
      Phase 1 – Assign sections found at the canonical heading level (most
                 common level among all matched sections).
      Phase 2 – For sections not found at the canonical level, pick the LAST
                 candidate within the expected TOC range (between the previous
                 and next already-placed section). Using the LAST avoids
                 picking brief intro sub-headings that appear before the real
                 section content.

    Returns a list of (sec_name, tag, idx) sorted by document position.
    """
    from collections import Counter

    upper = next_ch_pos if next_ch_pos > 0 else len(all_headings)

    # Collect all heading tags within the chapter range
    range_items = [(i, all_headings[i]) for i in range(ch_pos + 1, upper)]

    # Find ALL occurrences of each section name within the range
    section_names = [sec["name"] for sec in toc_sections]
    name_norms = [_normalize_heading(n) for n in section_names]
    norm_to_orig = dict(zip(name_norms, section_names))

    candidates = {n: [] for n in name_norms}
    for i, tag in range_items:
        text_norm = _normalize_heading(tag.get_text())
        if text_norm in candidates:
            candidates[text_norm].append((i, tag))

    # Determine canonical heading level using LAST candidate for each section.
    # The actual section heading always appears later than any brief intro with the same name,
    # so the last candidate reflects the true section level more reliably.
    last_levels = [int(cands[-1][1].name[1]) for cands in candidates.values() if cands]
    canonical_level = Counter(last_levels).most_common(1)[0][0] if last_levels else 3

    # Phase 1: Assign canonical-level matches (use first occurrence at canon level)
    placed = {}  # name_norm → (idx, tag) or None
    for nn in name_norms:
        cands = candidates[nn]
        at_canon = [(i, t) for i, t in cands if int(t.name[1]) == canonical_level]
        placed[nn] = at_canon[0] if at_canon else None

    # Phase 2: Fill in sections not found at canonical level
    # Process in TOC order so that earlier placements narrow later ranges
    for idx_in_list, nn in enumerate(name_norms):
        if placed[nn] is not None:
            continue

        # Determine search range: after last placed predecessor, before first
        # placed successor
        prev_pos = ch_pos
        for j in range(idx_in_list - 1, -1, -1):
            prev_nn = name_norms[j]
            if placed[prev_nn] is not None:
                prev_pos = placed[prev_nn][0]
                break

        next_pos = upper
        for j in range(idx_in_list + 1, len(name_norms)):
            next_nn = name_norms[j]
            if placed[next_nn] is not None:
                next_pos = placed[next_nn][0]
                break

        # Among candidates in (prev_pos, next_pos), pick the LAST one.
        # This avoids brief intro sub-headings that appear immediately after
        # the previous section heading — the actual content section is the
        # last occurrence before the next section boundary.
        in_range = [(i, t) for i, t in candidates[nn] if prev_pos < i < next_pos]
        if in_range:
            placed[nn] = in_range[-1]  # LAST in range

    # Build final sorted result
    result = []
    for nn, orig_name in zip(name_norms, section_names):
        p = placed[nn]
        if p is None:
            result.append((orig_name, None, -1))
        else:
            result.append((orig_name, p[1], p[0]))

    # Sort by document position (preserves reading order)
    result.sort(key=lambda x: x[2] if x[2] >= 0 else float("inf"))
    return result


def parse_html_named(html_path, toc_chapters, images_dir):
    """Parse HTML for named-format books (DDIA / O'Reilly style)."""
    print("   Loading HTML file...")
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Flat ordered list of ALL headings (h1–h5) for sequential search
    all_headings = soup.find_all(["h1", "h2", "h3", "h4", "h5"])
    print(f"   Found {len(all_headings)} total headings in HTML")

    # Pass 1: Locate every chapter heading (only h1/h2/h3 — never h4/h5)
    chapter_positions = []  # [(chapter, tag, idx)]
    search_from = 0
    for chapter in toc_chapters:
        tag, idx = _find_heading(
            chapter["name"],
            all_headings,
            search_from,
            strip_chapter_prefix=True,
            max_level=3,
        )
        chapter_positions.append((chapter, tag, idx))
        if idx >= 0:
            search_from = idx + 1

    found_count = sum(1 for _, t, _ in chapter_positions if t is not None)
    print(f"   Located {found_count} / {len(toc_chapters)} chapter headings")

    # Collect chapter heading ids for use as stop boundaries
    chapter_heading_ids = {id(t) for _, t, _ in chapter_positions if t is not None}

    image_counter = [1]
    result_chapters = []

    for ch_i, (chapter, ch_tag, ch_pos) in enumerate(chapter_positions):
        if ch_tag is None:
            print(f"\n   WARNING: Chapter '{chapter['name']}' not found — skipping")
            continue

        # Determine the range for this chapter (up to the next chapter heading)
        next_ch_pos = len(all_headings)
        for j in range(ch_i + 1, len(chapter_positions)):
            if chapter_positions[j][2] >= 0:
                next_ch_pos = chapter_positions[j][2]
                break

        print(f"\n   Chapter: {chapter['name']}")

        # Pass 2: Smart section matching within chapter range
        section_tags = _find_sections_for_chapter(
            chapter["sections"], all_headings, ch_pos, next_ch_pos
        )

        # Pass 3: Extract content for each section (stop at next section tag)
        result_sections = []
        for i, (sec_name, sec_tag, _) in enumerate(section_tags):
            if should_exclude_section(sec_name):
                print(f"     Skipping '{sec_name}' (excluded)")
                continue

            if sec_tag is None:
                print(f"     WARNING: Section '{sec_name}' not found in HTML")
                continue

            # stop_tag = next FOUND section tag in this chapter
            stop_tag = None
            for j in range(i + 1, len(section_tags)):
                if section_tags[j][1] is not None:
                    stop_tag = section_tags[j][1]
                    break

            content = _extract_until_tag(
                sec_tag, stop_tag, chapter_heading_ids, images_dir, image_counter
            )

            if content:
                result_sections.append({"name": sec_name, "content": content})
                print(f"     {sec_name}: {len(content)} chars")
            else:
                print(f"     '{sec_name}': WARNING - empty content")

        if result_sections:
            result_chapters.append(
                {"name": chapter["name"], "sections": result_sections}
            )

    return result_chapters


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def parse_html_with_toc(html_path, toc_result, images_dir):
    """Dispatch to numbered or named HTML parser based on book format."""
    book_format = toc_result["format"]
    toc_chapters = toc_result["chapters"]

    if book_format == "numbered":
        return parse_html_numbered(html_path, toc_chapters, images_dir)
    else:
        return parse_html_named(html_path, toc_chapters, images_dir)


def convert_book(book_dir):
    """Convert a book from its directory"""
    book_path = Path(book_dir)

    if not book_path.exists():
        print(f"Error: Directory {book_dir} not found")
        return

    pdf_path = book_path / "content.pdf"
    html_path = book_path / "content.html"

    if not pdf_path.exists():
        print(f"Error: content.pdf not found in {book_dir}")
        return
    if not html_path.exists():
        print(f"Error: content.html not found in {book_dir}")
        return

    images_dir = book_path / "images"
    images_dir.mkdir(exist_ok=True)

    print(f"\nProcessing book: {book_path.name}")
    print("=" * 60)

    # Step 1: Extract TOC from PDF
    print("\n1. Extracting TOC from PDF...")
    toc_result = extract_toc_from_pdf(pdf_path)
    toc_chapters = toc_result["chapters"]
    book_format = toc_result["format"]
    print(f"   Format: {book_format}")
    print(f"   Found {len(toc_chapters)} chapters")

    if not toc_chapters:
        print("Error: Could not extract TOC from PDF")
        return

    # Step 2: Parse HTML using TOC
    print("\n2. Parsing HTML content using TOC...")
    chapters = parse_html_with_toc(html_path, toc_result, images_dir)

    # Step 3: Create JSON output
    print("\n\n3. Creating JSON output...")
    output_data = {"chapters": chapters}

    output_path = book_path / "content.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    total_sections = sum(len(ch["sections"]) for ch in chapters)
    total_chars = sum(len(s["content"]) for ch in chapters for s in ch["sections"])

    print(f"\n{'=' * 60}")
    print(f"Successfully created {output_path}")
    print(f"  Chapters: {len(chapters)}")
    print(f"  Sections: {total_sections}")
    print(f"  Total content: {total_chars:,} characters")
    print(f"  Images saved to: {images_dir}")


if __name__ == "__main__":
    book_dir = sys.argv[1] if len(sys.argv) > 1 else "books/Computer Architecture"
    convert_book(book_dir)
