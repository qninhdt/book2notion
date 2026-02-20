# notion2book

A three-step pipeline to turn a scanned/OCR'd book into structured Notion notes:

1. **convert** — Extract the TOC from the PDF and map it to the OCR output, producing `content.json`
2. **generate** — Call an LLM to summarise each chapter section into structured JSON + Markdown
3. **sync** — Push the generated chapters to a Notion workspace

## Project structure

```text
books/
  <Book Name>.pdf

outputs/
  <Book Name>/
    hybrid_auto/
      <Book Name>_content_list_v2.json   # produced by: mineru (OCR, step 0)
      images/...
    content.json                          # produced by: convert
    images/
      figure_0001.jpg                     # normalized images copied from hybrid_auto/
    chapters/
      1. Chapter Name.json                # produced by: generate
      1. Chapter Name.md

notion2book/
  __init__.py
  __main__.py
  cli.py
  project.py
  convert.py
  generate.py
  sync.py

prompts/
  summarize.md
```

## Install

```bash
pip install -r requirements.txt
```

## Environment

Create `.env` from the provided example:

```bash
cp .env.example .env
```

Then fill in your values:

```dotenv
# LLM
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_ID=gpt-4o

# Notion
NOTION_API_KEY=secret_...
NOTION_PAGE_ID=...
```

## Usage

### 0. OCR (pre-requisite)

Before running the pipeline, each book PDF must be processed with [MinerU](https://github.com/opendatalab/MinerU) to produce the OCR JSON. A ready-to-run notebook is provided at `notebooks/ocr.ipynb`.

```bash
pip install -U "mineru[all]"
mineru -p books -o outputs
```

This scans every PDF under `books/` and writes the OCR results into `outputs/<Book Name>/hybrid_auto/`.

---

Run via the module entrypoint:

```bash
python -m notion2book <command> [options]
```

### 1. convert

Builds `outputs/<Book Name>/content.json` from the PDF TOC and OCR data.

```bash
# One book
python -m notion2book convert --book "Designing Data-Intensive Applications"

# All books that have complete inputs
python -m notion2book convert
```

Inputs:
- `books/<Book Name>.pdf`
- `outputs/<Book Name>/hybrid_auto/<Book Name>_content_list_v2.json`

Output:
- `outputs/<Book Name>/content.json`
- `outputs/<Book Name>/images/figure_XXXX.*` (normalised image copies)

### 2. generate

Calls an LLM to summarise each chapter and writes JSON + Markdown files.

```bash
# All chapters for one book
python -m notion2book generate --book "Designing Data-Intensive Applications"

# Specific chapters
python -m notion2book generate --book "Designing Data-Intensive Applications" --chapter 1 --chapter 3

# All books that have content.json
python -m notion2book generate
```

Output: `outputs/<Book Name>/chapters/*.json` and `*.md`

### 3. sync

Pushes generated chapters to Notion.

```bash
# All books
python -m notion2book sync

# One book
python -m notion2book sync --book "Designing Data-Intensive Applications"
```

Reads: `outputs/<Book Name>/chapters/*.json`  
Images: resolved from `outputs/<Book Name>/images/`

## Notes

- Run `convert` before `generate` if the OCR output changes.
- Run `generate` before `sync` if chapter summaries change.
- `generate` skips chapters whose `.json` already exists (resume-safe).
- `sync` uses a file-hash cache (`.sync_cache.json`) to skip unchanged chapters.
