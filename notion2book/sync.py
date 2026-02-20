#!/usr/bin/env python3
"""Sync generated book notes to a Notion workspace.

Creates a hierarchical structure:
  Parent Page → Books Database → Book Page → Chapters Database (inline) → Chapter Page

Each chapter page renders the structured JSON output as rich Notion blocks.

Environment variables (in .env):
    NOTION_API_KEY   - Notion integration token
    NOTION_PAGE_ID   - Parent page ID for the books database
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from notion_client import Client
from notion_client.errors import APIResponseError

from .project import OUTPUTS_ROOT, list_output_book_dirs

BATCH_SIZE = 100

# Module-level client, initialized in run()
notion: Client

# ---------------------------------------------------------------------------
# Notion API helpers
# ---------------------------------------------------------------------------


def api(fn, *a, **kw):
    for attempt in range(6):
        try:
            return fn(*a, **kw)
        except APIResponseError as e:
            if e.status == 429:
                wait = float((e.headers or {}).get("Retry-After", 2**attempt))
                print(f"  Rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
            elif attempt < 5:
                time.sleep(min(2**attempt, 30))
            else:
                raise


def append_blocks(page_id, blocks):
    for i in range(0, len(blocks), BATCH_SIZE):
        api(
            notion.blocks.children.append,
            block_id=page_id,
            children=blocks[i : i + BATCH_SIZE],
        )
        if i + BATCH_SIZE < len(blocks):
            time.sleep(0.35)


def list_children(block_id):
    results, cursor = [], None
    while True:
        kw = {"block_id": block_id}
        if cursor:
            kw["start_cursor"] = cursor
        r = api(notion.blocks.children.list, **kw)
        results.extend(r["results"])
        if not r.get("has_more"):
            break
        cursor = r["next_cursor"]
    return results


def get_data_source_id(db_id):
    db = api(notion.databases.retrieve, database_id=db_id)
    ds_list = db.get("data_sources", [])
    return ds_list[0]["id"] if ds_list else db_id


def find_page(ds_id, title):
    r = api(
        notion.data_sources.query,
        data_source_id=ds_id,
        filter={"property": "Name", "title": {"equals": title}},
    )
    return r["results"][0]["id"] if r["results"] else None


def normalize_page_id(pid):
    if pid and pid.startswith("http"):
        cleaned = pid.split("?")[0].split("#")[0]
        match = re.search(r"([a-f0-9]{32})", cleaned.replace("-", ""))
        if match:
            h = match.group(1)
            return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"
    return pid


# ---------------------------------------------------------------------------
# Sync cache
# ---------------------------------------------------------------------------


def _file_hash(path):
    """SHA-1 of the raw file bytes."""
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _load_cache(out_dir):
    cache_file = Path(out_dir) / ".sync_cache.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass
    return {}


def _save_cache(out_dir, cache):
    cache_file = Path(out_dir) / ".sync_cache.json"
    cache_file.write_text(json.dumps(cache, indent=2))


# ---------------------------------------------------------------------------
# Image upload
# ---------------------------------------------------------------------------

_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}


def upload_image(filepath):
    """Upload a local image to Notion via the File Uploads API (single-part)."""
    mime = _MIME.get(filepath.suffix.lower(), "image/jpeg")
    name = filepath.name

    upload = api(
        notion.file_uploads.create,
        mode="single_part",
        filename=name,
        content_type=mime,
    )
    upload_id = upload["id"]

    with open(filepath, "rb") as f:
        api(
            notion.file_uploads.send,
            file_upload_id=upload_id,
            file=(name, f, mime),
        )

    return upload_id


# ---------------------------------------------------------------------------
# Rich-text primitives
# ---------------------------------------------------------------------------


def _rt(content, bold=False, italic=False, code=False, color="default", link=None):
    obj = {
        "type": "text",
        "text": {"content": content},
        "annotations": {
            "bold": bold,
            "italic": italic,
            "code": code,
            "strikethrough": False,
            "underline": False,
            "color": color,
        },
    }
    if link:
        obj["text"]["link"] = {"url": link}
    return obj


def _eq(expr):
    return {"type": "equation", "equation": {"expression": expr.strip()}}


def _chunk(content, limit=2000, **kw):
    out = []
    while content:
        out.append(_rt(content[:limit], **kw))
        content = content[limit:]
    return out


# ---------------------------------------------------------------------------
# Inline markdown → rich_text[]
# ---------------------------------------------------------------------------


def parse_inline(text):
    if not text:
        return []

    result = []
    buf = ""
    bold = False
    italic = False
    n = len(text)
    i = 0

    def flush():
        nonlocal buf
        if buf:
            result.extend(_chunk(buf, bold=bold, italic=italic))
            buf = ""

    while i < n:
        c = text[i]

        if c == "`":
            flush()
            end = text.find("`", i + 1)
            if end < 0:
                buf += c
                i += 1
                continue
            result.extend(_chunk(text[i + 1 : end], code=True))
            i = end + 1
            continue

        if c == "$":
            if text[i : i + 2] == "$$":
                end = text.find("$$", i + 2)
                if end >= 0:
                    flush()
                    result.append(_eq(text[i + 2 : end]))
                    i = end + 2
                    continue
            end = text.find("$", i + 1)
            if end >= 0 and "\n" not in text[i + 1 : end] and end > i + 1:
                flush()
                result.append(_eq(text[i + 1 : end]))
                i = end + 1
                continue
            buf += c
            i += 1
            continue

        three = text[i : i + 3]
        two = text[i : i + 2]

        if three in ("***", "___"):
            flush()
            bold = not bold
            italic = not italic
            i += 3
            continue
        if two in ("**", "__"):
            flush()
            bold = not bold
            i += 2
            continue
        if c == "*":
            flush()
            italic = not italic
            i += 1
            continue
        if c == "_":
            pa = i > 0 and text[i - 1].isalnum()
            na = i + 1 < n and text[i + 1].isalnum()
            if not (pa and na):
                flush()
                italic = not italic
                i += 1
                continue

        if c == "[":
            m = re.match(r"\[([^\]]+)\]\(([^\)]+)\)", text[i:])
            if m:
                flush()
                result.extend(
                    _chunk(m.group(1), bold=bold, italic=italic, link=m.group(2))
                )
                i += m.end()
                continue

        buf += c
        i += 1

    flush()
    return result or [_rt("")]


# ---------------------------------------------------------------------------
# Code-language normalisation
# ---------------------------------------------------------------------------

_LANG_MAP = {
    "cpp": "c++",
    "cc": "c++",
    "cxx": "c++",
    "c++": "c++",
    "csharp": "c#",
    "cs": "c#",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "py": "python",
    "python3": "python",
    "rb": "ruby",
    "sh": "shell",
    "zsh": "shell",
    "bash": "bash",
    "yml": "yaml",
    "md": "markdown",
    "dockerfile": "docker",
    "objc": "objective-c",
    "txt": "plain text",
    "text": "plain text",
    "plaintext": "plain text",
    "": "plain text",
}

_VALID_LANGS = {
    "abap",
    "abc",
    "agda",
    "arduino",
    "ascii art",
    "assembly",
    "bash",
    "basic",
    "bnf",
    "c",
    "c#",
    "c++",
    "clojure",
    "coffeescript",
    "coq",
    "css",
    "dart",
    "dhall",
    "diff",
    "docker",
    "ebnf",
    "elixir",
    "elm",
    "erlang",
    "f#",
    "flow",
    "fortran",
    "gherkin",
    "glsl",
    "go",
    "graphql",
    "groovy",
    "haskell",
    "hcl",
    "html",
    "idris",
    "java",
    "javascript",
    "json",
    "julia",
    "kotlin",
    "latex",
    "less",
    "lisp",
    "livescript",
    "llvm ir",
    "lua",
    "makefile",
    "markdown",
    "markup",
    "matlab",
    "mathematica",
    "mermaid",
    "nix",
    "notion formula",
    "objective-c",
    "ocaml",
    "pascal",
    "perl",
    "php",
    "plain text",
    "powershell",
    "prolog",
    "protobuf",
    "purescript",
    "python",
    "r",
    "racket",
    "reason",
    "ruby",
    "rust",
    "sass",
    "scala",
    "scheme",
    "scss",
    "shell",
    "smalltalk",
    "solidity",
    "sql",
    "swift",
    "toml",
    "typescript",
    "vb.net",
    "verilog",
    "vhdl",
    "visual basic",
    "webassembly",
    "xml",
    "yaml",
    "java/c/c++/c#",
}


def _lang(s):
    key = (s or "").strip().lower()
    mapped = _LANG_MAP.get(key, key) or "plain text"
    return mapped if mapped in _VALID_LANGS else "plain text"


# ---------------------------------------------------------------------------
# Block-level markdown → Notion blocks
# ---------------------------------------------------------------------------


def _is_table_row(line):
    return bool(line.strip().startswith("|") and line.strip().endswith("|"))


def _parse_table(lines, i):
    """Parse a GFM markdown table into a Notion table block."""
    rows = []
    n = len(lines)
    while i < n and _is_table_row(lines[i]):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        rows.append(cells)
        i += 1
    # Remove separator row (contains only dashes and colons)
    rows = [r for r in rows if not all(re.match(r"^:?-+:?$", c) for c in r if c)]
    if not rows:
        return i, []
    col_count = max(len(r) for r in rows)
    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "type": "table_row",
                "table_row": {
                    "cells": [
                        parse_inline(row[ci] if ci < len(row) else "")
                        for ci in range(col_count)
                    ]
                },
            }
        )
    block = {
        "type": "table",
        "table": {
            "table_width": col_count,
            "has_column_header": True,
            "has_row_header": False,
            "children": table_rows,
        },
    }
    return i, [block]


def _is_block_start(line):
    s = line.strip()
    return bool(
        s.startswith("#")
        or re.match(r"^[-*]\s", s)
        or re.match(r"^\d+\.\s", s)
        or s.startswith("> ")
        or s.startswith("```")
        or s.startswith("$$")
        or re.match(r"^---+$", s)
        or s.startswith("![")
        or _is_table_row(s)
    )


def md_to_blocks(text, book_dir=None):
    if not text:
        return []
    blocks = []
    lines = text.split("\n")
    i, n = 0, len(lines)

    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue

        if s.startswith("$$"):
            if s == "$$":
                parts = []
                i += 1
                while i < n and lines[i].strip() != "$$":
                    parts.append(lines[i])
                    i += 1
                if i < n:
                    i += 1
                expr = "\n".join(parts).strip()
            else:
                expr = s[2:]
                if expr.endswith("$$"):
                    expr = expr[:-2]
                expr = expr.strip()
                i += 1
            blocks.append({"type": "equation", "equation": {"expression": expr}})
            continue

        if s.startswith("```"):
            lg = s[3:].strip()
            cl = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                cl.append(lines[i])
                i += 1
            if i < n:
                i += 1
            blocks.append(
                {
                    "type": "code",
                    "code": {
                        "rich_text": _chunk("\n".join(cl)),
                        "language": _lang(lg),
                    },
                }
            )
            continue

        hm = re.match(r"^(#{1,3})\s+(.+)$", s)
        if hm:
            lv = f"heading_{min(len(hm.group(1)), 3)}"
            blocks.append({"type": lv, lv: {"rich_text": parse_inline(hm.group(2))}})
            i += 1
            continue

        if re.match(r"^---+$", s):
            blocks.append({"type": "divider", "divider": {}})
            i += 1
            continue

        im = re.match(r"^!\[([^\]]*)\]\(([^\)]+)\)", s)
        if im:
            blocks.append(_build_md_image(im.group(2), im.group(1), book_dir))
            i += 1
            continue

        if _is_table_row(s):
            i, tblocks = _parse_table(lines, i)
            blocks.extend(tblocks)
            continue

        if s.startswith("> "):
            ql = []
            while i < n and lines[i].strip().startswith("> "):
                ql.append(lines[i].strip()[2:])
                i += 1
            blocks.append(
                {"type": "quote", "quote": {"rich_text": parse_inline("\n".join(ql))}}
            )
            continue

        if re.match(r"^[-*]\s", s):
            i, lb = _parse_list(lines, i, bullet=True)
            blocks.extend(lb)
            continue

        if re.match(r"^\d+\.\s", s):
            i, lb = _parse_list(lines, i, bullet=False)
            blocks.extend(lb)
            continue

        pl = [s]
        i += 1
        while i < n and lines[i].strip() and not _is_block_start(lines[i]):
            pl.append(lines[i].strip())
            i += 1
        blocks.append(
            {
                "type": "paragraph",
                "paragraph": {"rich_text": parse_inline(" ".join(pl))},
            }
        )

    return blocks


def _resolve_local_image(image_ref, book_dir):
    """Resolve a markdown image reference from outputs/{book}/images."""
    if not image_ref:
        return None
    if re.match(r"^https?://", image_ref):
        return None

    rel = image_ref.lstrip("./")
    if rel.startswith("images/"):
        rel = rel[len("images/") :]

    primary = book_dir / "images" / rel
    if primary.exists() and primary.is_file():
        return primary

    # Allow extension mismatch in markdown (e.g. figure_0001.png vs .jpg on disk).
    stem = Path(rel).stem
    if stem.startswith("figure_"):
        for ext in (".png", ".jpeg", ".jpg", ".gif", ".webp"):
            candidate = book_dir / "images" / f"{stem}{ext}"
            if candidate.exists() and candidate.is_file():
                return candidate

    return None


def _build_md_image(path, caption="", book_dir=None):
    caption_rt = parse_inline(caption)[:25] if caption else []

    if book_dir is not None:
        local = _resolve_local_image(path, book_dir)
        if local is not None:
            try:
                uid = upload_image(local)
                return {
                    "type": "image",
                    "image": {
                        "type": "file_upload",
                        "file_upload": {"id": uid},
                        "caption": caption_rt,
                    },
                }
            except Exception as e:
                print(f"      [warn] Upload failed {local.name}: {e}")

    if re.match(r"^https?://", path):
        return {
            "type": "image",
            "image": {
                "type": "external",
                "external": {"url": path},
                "caption": caption_rt,
            },
        }

    return {
        "type": "callout",
        "callout": {
            "rich_text": [_rt(f"Missing image: {path}", italic=True)],
            "icon": {"emoji": "\U0001f5bc\ufe0f"},
        },
    }


def _parse_list(lines, i, bullet=True):
    pat = r"^[-*]\s" if bullet else r"^\d+\.\s"
    btype = "bulleted_list_item" if bullet else "numbered_list_item"
    strip_re = r"^[-*]\s+" if bullet else r"^\d+\.\s+"
    items = []
    n = len(lines)

    while i < n:
        raw = lines[i]
        s = raw.strip()
        indent = len(raw) - len(raw.lstrip())
        if not s:
            if i + 1 < n and _is_list_continuation(lines[i + 1], pat):
                i += 1
                continue
            break
        if indent < 2 and re.match(pat, s):
            items.append({"t": re.sub(strip_re, "", s, count=1), "ch": []})
            i += 1
        elif indent >= 2 and items:
            if re.match(r"^[-*]\s", s):
                items[-1]["ch"].append(("b", re.sub(r"^[-*]\s+", "", s, count=1)))
                i += 1
            elif re.match(r"^\d+\.\s", s):
                items[-1]["ch"].append(("n", re.sub(r"^\d+\.\s+", "", s, count=1)))
                i += 1
            else:
                items[-1]["t"] += " " + s
                i += 1
        else:
            break

    blocks = []
    for it in items:
        b = {"type": btype, btype: {"rich_text": parse_inline(it["t"])}}
        if it["ch"]:
            children = []
            for ct, ctxt in it["ch"]:
                ctype = "bulleted_list_item" if ct == "b" else "numbered_list_item"
                children.append(
                    {"type": ctype, ctype: {"rich_text": parse_inline(ctxt)}}
                )
            b[btype]["children"] = children
        blocks.append(b)
    return i, blocks


def _is_list_continuation(line, pat):
    s = line.strip()
    return bool(re.match(pat, s)) or (len(line) - len(line.lstrip()) >= 2 and bool(s))


def _flatten_children(blocks):
    """Remove nested children to stay within Notion's 2-level depth limit."""
    out = []
    for b in blocks:
        bt = b["type"]
        if bt in ("bulleted_list_item", "numbered_list_item"):
            ch = b[bt].pop("children", None)
            out.append(b)
            if ch:
                out.extend(ch)
        else:
            out.append(b)
    return out


# ---------------------------------------------------------------------------
# Section / chapter → Notion blocks
# ---------------------------------------------------------------------------


def build_section(section, sec_idx, book_dir):
    blocks = []

    blocks.append(
        {
            "type": "heading_1",
            "heading_1": {"rich_text": [_rt(f"{sec_idx}. {section['name']}")]},
        }
    )

    if section.get("summary"):
        blocks.append(
            {
                "type": "paragraph",
                "paragraph": {
                    "rich_text": parse_inline(section["summary"]),
                    "color": "gray",
                },
            }
        )

    for si, sub in enumerate(section.get("subsections") or [], 1):
        blocks.append(
            {
                "type": "heading_2",
                "heading_2": {"rich_text": [_rt(f"{sec_idx}.{si} {sub['name']}")]},
            }
        )
        blocks.extend(md_to_blocks(sub.get("content", ""), book_dir))
        for fig in sub.get("figures") or []:
            blocks.append(_build_figure(fig, book_dir))

    if section.get("code") and section["code"].get("content"):
        c = section["code"]
        blocks.append(
            {
                "type": "code",
                "code": {
                    "rich_text": _chunk(c["content"]),
                    "language": _lang(c.get("lang", "")),
                },
            }
        )

    if section.get("interview"):
        children = []
        for qa in section["interview"]:
            question = qa.get("question") or ""
            answer = qa.get("answer") or ""
            if not question and not answer:
                continue
            q_rt = [_rt("Question: ", bold=True)] + parse_inline(question)
            if qa.get("level"):
                q_rt.append(_rt(f" ({qa['level']})", italic=True, color="gray"))
            # quote and answer toggle are siblings (Notion disallows children on a
            # toggle that is itself inside quote.children — 3-level nesting limit)
            children.append({"type": "quote", "quote": {"rich_text": q_rt}})
            children.append(
                {
                    "type": "toggle",
                    "toggle": {
                        "rich_text": [_rt("Answer", bold=True)],
                        "children": [
                            {
                                "type": "paragraph",
                                "paragraph": {
                                    "rich_text": parse_inline(answer),
                                },
                            }
                        ],
                    },
                }
            )
        blocks.append({"type": "paragraph", "paragraph": {"rich_text": [_rt("")]}})
        blocks.append(
            {
                "type": "toggle",
                "toggle": {
                    "rich_text": [_rt("Interview", bold=True, italic=True)],
                    "color": "blue_background",
                    "children": children,
                },
            }
        )

    if section.get("more"):
        children = []
        for m in section["more"]:
            children.append(
                {
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [_rt(m["name"], bold=True)],
                    },
                }
            )
            children.extend(
                _flatten_children(md_to_blocks(m.get("content", ""), book_dir))
            )
        blocks.append(
            {
                "type": "toggle",
                "toggle": {
                    "rich_text": [_rt("More", bold=True, italic=True)],
                    "color": "yellow_background",
                    "children": children,
                },
            }
        )

    retained = section.get("retained") or []
    omitted = section.get("omitted") or []
    if retained or omitted:
        children = []
        if retained:
            children.append(
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [_rt("Retained", bold=True)],
                    },
                }
            )
            for item in retained:
                children.append(
                    {
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [
                                _rt(item["name"], bold=True),
                                _rt(f" — {item['reason']}"),
                            ],
                        },
                    }
                )
        if omitted:
            children.append(
                {
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [_rt("Omitted", bold=True)],
                    },
                }
            )
            for item in omitted:
                children.append(
                    {
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [
                                _rt(item["name"], bold=True),
                                _rt(f" — {item['reason']}"),
                            ],
                        },
                    }
                )
        blocks.append(
            {
                "type": "toggle",
                "toggle": {
                    "rich_text": [_rt("Editorial", bold=True, italic=True)],
                    "color": "gray_background",
                    "children": children,
                },
            }
        )

    blocks.append({"type": "divider", "divider": {}})
    return blocks


def _build_figure(fig, book_dir):
    caption = fig.get("caption", "")
    fid = fig.get("id")
    caption_rt = parse_inline(caption)[:25] if caption else []

    if fid is not None and book_dir is not None:
        for ext in (".png", ".jpeg", ".jpg", ".gif", ".webp"):
            img_path = book_dir / "images" / f"figure_{fid:04d}{ext}"
            if img_path.exists():
                try:
                    uid = upload_image(img_path)
                    return {
                        "type": "image",
                        "image": {
                            "type": "file_upload",
                            "file_upload": {"id": uid},
                            "caption": caption_rt,
                        },
                    }
                except Exception as e:
                    print(f"      [warn] Upload failed {img_path.name}: {e}")
                break

    parts = [_rt("Figure: ", bold=True, italic=True)]
    if caption:
        parts.extend(_chunk(caption[:1800], italic=True))
    return {
        "type": "callout",
        "callout": {"rich_text": parts, "icon": {"emoji": "\U0001f5bc\ufe0f"}},
    }


# ---------------------------------------------------------------------------
# Database management
# ---------------------------------------------------------------------------


def get_or_create_books_db(parent_id):
    """Find or create the top-level Books database. Returns (db_id, ds_id)."""
    for blk in list_children(parent_id):
        if blk["type"] == "child_database":
            db_id = blk["id"]
            db = api(notion.databases.retrieve, database_id=db_id)
            title_text = "".join(t.get("plain_text", "") for t in db.get("title", []))
            if "Books" in title_text:
                return db_id, get_data_source_id(db_id)

    db = api(
        notion.request,
        path="databases",
        method="POST",
        body={
            "parent": {"type": "page_id", "page_id": parent_id},
            "title": [{"type": "text", "text": {"content": "Books"}}],
            "icon": {"emoji": "\U0001f4da"},
            "initial_data_source": {
                "properties": {"Name": {"title": {}}},
            },
        },
    )
    return db["id"], db["data_sources"][0]["id"]


def get_or_create_chapters_db(book_page_id):
    """Find or create the inline Chapters database inside a book page. Returns (db_id, ds_id)."""
    for blk in list_children(book_page_id):
        if blk["type"] == "child_database":
            db_id = blk["id"]
            return db_id, get_data_source_id(db_id)

    db = api(
        notion.request,
        path="databases",
        method="POST",
        body={
            "parent": {"type": "page_id", "page_id": book_page_id},
            "title": [{"type": "text", "text": {"content": "Chapters"}}],
            "is_inline": True,
            "initial_data_source": {
                "properties": {
                    "Name": {"title": {}},
                    "Chapter": {"number": {"format": "number"}},
                },
            },
        },
    )
    return db["id"], db["data_sources"][0]["id"]


# ---------------------------------------------------------------------------
# Sync logic
# ---------------------------------------------------------------------------


def _ch_sort_key(path):
    """Extract numeric chapter number for sorting."""
    m = re.match(r"^(\d+)", path.stem)
    return int(m.group(1)) if m else 999


def sync_chapter(ch_ds_id, path, book_dir, ch_num, cache):
    title = path.stem
    new_hash = _file_hash(path)

    if cache.get(title) == new_hash:
        print(f"    [skip] {title}")
        return

    existing = find_page(ch_ds_id, title)
    if existing:
        # Archive old page and recreate — faster than deleting blocks one by one
        api(notion.pages.update, page_id=existing, archived=True)

    with open(path) as f:
        data = json.load(f)

    blocks = []
    for si, sec in enumerate(data.get("sections", []), 1):
        blocks.extend(build_section(sec, si, book_dir))

    page = api(
        notion.pages.create,
        parent={"type": "data_source_id", "data_source_id": ch_ds_id},
        properties={
            "Name": {"title": [{"text": {"content": title}}]},
            "Chapter": {"number": ch_num},
        },
    )
    append_blocks(page["id"], blocks)
    cache[title] = new_hash
    print(f"    [{'update' if existing else 'new'}] {title}")


def sync_book(books_ds_id, book_dir):
    name = book_dir.name
    chapters_dir = book_dir / "chapters"
    if not chapters_dir.exists():
        return

    jsons = sorted(
        (p for p in chapters_dir.glob("*.json") if not p.name.startswith(".")),
        key=_ch_sort_key,
    )
    if not jsons:
        return

    print(f"\n  Book: {name} ({len(jsons)} chapter(s))")

    page_id = find_page(books_ds_id, name)
    if not page_id:
        page = api(
            notion.pages.create,
            parent={"type": "data_source_id", "data_source_id": books_ds_id},
            properties={"Name": {"title": [{"text": {"content": name}}]}},
        )
        page_id = page["id"]
        print(f"    Created book entry")

    _, ch_ds_id = get_or_create_chapters_db(page_id)
    cache = _load_cache(chapters_dir)

    for jf in jsons:
        ch_num = _ch_sort_key(jf)
        try:
            sync_chapter(ch_ds_id, jf, book_dir, ch_num, cache)
        except Exception as e:
            print(f"    [error] {jf.stem}: {e}")

    _save_cache(chapters_dir, cache)


def run(book=None):
    global notion

    load_dotenv()

    api_key = os.getenv("NOTION_API_KEY")
    page_id_raw = os.getenv("NOTION_PAGE_ID")

    if not api_key or not page_id_raw:
        print("Error: Set NOTION_API_KEY and NOTION_PAGE_ID in .env")
        return 1
    if not OUTPUTS_ROOT.exists():
        print(f"Error: outputs directory not found: {OUTPUTS_ROOT}")
        return 1

    page_id = normalize_page_id(page_id_raw)
    notion = Client(auth=api_key)

    print("Connecting to Notion...")
    _, books_ds_id = get_or_create_books_db(page_id)
    print(f"Books database ready")

    matched = 0
    for d in list_output_book_dirs(book):
        if d.is_dir():
            matched += 1
            sync_book(books_ds_id, d)
    if book and matched == 0:
        print(f"Error: book not found under outputs/: {book}")
        return 1

    print("\nSync complete!")
    return 0
