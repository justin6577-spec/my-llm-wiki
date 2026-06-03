"""
build_wiki.py — PDF → wiki note converter
Reads unprocessed PDFs from raw/, extracts text with PyMuPDF,
sends to Claude to generate a structured Obsidian note,
and writes the result to wiki/.

Usage:
    python3 build_wiki.py          # process all unmatched PDFs
    python3 build_wiki.py Mamba.pdf  # process one specific PDF
"""

import anthropic
import os
import re
import sys
from pathlib import Path

# Wiki location: override with WIKI_VAULT env var; defaults to this file's directory.
VAULT = Path(os.environ.get("WIKI_VAULT", Path(__file__).resolve().parent))
RAW   = VAULT / "raw"
WIKI  = VAULT / "wiki"


def _strip_md_fence(text: str) -> str:
    """Strip leading/trailing markdown code fences that the API
    sometimes wraps responses in.

    Handles ```markdown ... ```, ```md ... ```, and bare ``` ... ```.
    """
    text = text.strip()
    # Remove opening fence (```markdown, ```md, ```)
    text = re.sub(r'^```(?:markdown|md|)[\r\n]+', '', text, flags=re.IGNORECASE)
    # Remove closing fence
    text = re.sub(r'[\r\n]+```\s*$', '', text)
    return text.strip() + '\n'

_PROJECT = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "your-gcp-project-id")
_REGION  = os.environ.get("VERTEX_REGION_CLAUDE_4_6_SONNET", "europe-west1")
_MODEL   = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet-4-6")

client = anthropic.AnthropicVertex(project_id=_PROJECT, region=_REGION)

SYSTEM = """\
You are an expert AI/ML research writer. You write Obsidian wiki notes in the style of
Andrej Karpathy: intuition first, concrete numbers, math only after the concept is clear.

Given extracted text from a research paper, produce a complete wiki note in this exact format:

```markdown
---
title: <Full Paper Title>
authors: <Author1, Author2, ...>
year: <YYYY>
arxiv: <arXiv ID if identifiable, else omit>
tags: [<tag1>, <tag2>, ...]
citation_count: 0
tldr: <One crisp sentence: what problem, what solution, key number>
---

## The Problem

<2–3 paragraphs: what existing approach was broken or slow, concrete failure mode>

## The Idea

<The core insight in one sentence, then unpack it intuitively>

## How It Works

<Key mechanism explained step by step — use analogies, then math>

## Key Results

<Concrete benchmarks: X× speedup, Y% improvement, compared to what baseline>

## Limitations

<What doesn't it solve? Where does it break down?>

## Why It Matters

<How does it connect to the broader LLM ecosystem? What did it unlock?>

## See Also

[[Transformer]] · [[Attention Is All You Need]] · <other relevant [[wikilinks]]>
```

Rules:
- Tags must be lowercase hyphenated: foundational, ssm, hardware, inference, attention, moe, glossary, benchmarks
- tldr must be a single sentence with at least one concrete number
- wikilinks use [[Title Case]] matching existing wiki note titles
- Do NOT wrap your output in a code block — output raw markdown only
- If the text is truncated or unclear, do your best with available information
"""

NOTE_PROMPT = """\
Paper text (may be truncated to first ~12 000 characters):

{text}

Write the complete wiki note for this paper.
"""


def extract_pdf_text(pdf_path: Path, max_chars: int = 12_000) -> str:
    """Extract text from a PDF using PyMuPDF, stopping at max_chars."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        parts = []
        total = 0
        for page in doc:
            chunk = page.get_text()
            parts.append(chunk)
            total += len(chunk)
            if total >= max_chars:
                break
        doc.close()
        return "".join(parts)[:max_chars]
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed for {pdf_path.name}: {e}") from e


def stem_to_wiki_title(stem: str) -> str:
    """Convert a filename stem to a plausible wiki note title for duplicate checking."""
    return stem.replace("_", " ").replace("-", " ")


def already_in_wiki(pdf_path: Path) -> bool:
    """Return True if a matching wiki note already exists for this PDF."""
    stem_lower = pdf_path.stem.lower().replace("_", "").replace("-", "").replace(" ", "")
    for note in WIKI.glob("*.md"):
        note_lower = note.stem.lower().replace("_", "").replace("-", "").replace(" ", "")
        if stem_lower in note_lower or note_lower in stem_lower:
            return True
    return False


def generate_note(pdf_path: Path) -> str:
    """Call Claude to generate a wiki note from the PDF text."""
    text = extract_pdf_text(pdf_path)
    if not text.strip():
        raise RuntimeError(f"No text extracted from {pdf_path.name}")

    response = client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": NOTE_PROMPT.format(text=text)}],
    )
    return _strip_md_fence(response.content[0].text)


def note_filename_from_content(content: str, fallback: str) -> str:
    """Extract the title from YAML frontmatter to use as the filename."""
    m = re.search(r"^title:\s*(.+)$", content, re.MULTILINE)
    if m:
        title = m.group(1).strip().strip('"').strip("'")
        # Sanitise for filesystem
        title = re.sub(r'[<>:"/\\|?*]', "", title)
        return title + ".md"
    return fallback + ".md"


def process_pdf(pdf_path: Path) -> dict:
    """Process one PDF: extract → generate → write. Returns status dict."""
    if already_in_wiki(pdf_path):
        return {"pdf": pdf_path.name, "status": "skipped", "reason": "already in wiki"}

    print(f"  Processing {pdf_path.name}…")
    try:
        content  = generate_note(pdf_path)
        filename = note_filename_from_content(content, pdf_path.stem)
        out_path = WIKI / filename

        if out_path.exists():
            return {"pdf": pdf_path.name, "status": "skipped", "reason": f"{filename} already exists"}

        out_path.write_text(content, encoding="utf-8")
        print(f"    ✓ wrote {filename}")
        return {"pdf": pdf_path.name, "status": "written", "path": str(out_path)}
    except Exception as e:
        print(f"    ✗ {e}")
        return {"pdf": pdf_path.name, "status": "error", "error": str(e)}


def main():
    if len(sys.argv) > 1:
        # Process specific PDFs named on the command line
        targets = [RAW / arg for arg in sys.argv[1:]]
    else:
        # Process every PDF in raw/ that lacks a matching wiki note
        targets = [p for p in sorted(RAW.glob("*.pdf")) if not already_in_wiki(p)]

    if not targets:
        print("build_wiki.py: nothing to process.")
        return

    print(f"build_wiki.py: processing {len(targets)} PDF(s)…")
    results = [process_pdf(p) for p in targets]

    written  = [r for r in results if r["status"] == "written"]
    skipped  = [r for r in results if r["status"] == "skipped"]
    errors   = [r for r in results if r["status"] == "error"]

    print(f"\nDone — {len(written)} written, {len(skipped)} skipped, {len(errors)} errors")
    for r in errors:
        print(f"  ✗ {r['pdf']}: {r.get('error', '?')}")


if __name__ == "__main__":
    main()
