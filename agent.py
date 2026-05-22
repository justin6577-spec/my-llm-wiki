"""
agent.py — Agentic wiki ingestion system
Uses Claude (via Vertex AI) with tool use to find, evaluate,
and add content to the LLM wiki from multiple sources.

Usage:
    python3 agent.py daily              # check arXiv + GitHub last 24 h
    python3 agent.py topic <query>      # add content about a topic
    python3 agent.py citations          # track high-impact citing papers
"""

import anthropic
import json
import os
import re
import sys
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

VAULT = Path("/work/HHRI-AI/Saqlain/my-wiki")
RAW   = VAULT / "raw"
WIKI  = VAULT / "wiki"
LOG   = VAULT / "agent_log.json"

WIKI_THEMES = [
    "transformer", "attention", "state space model", "SSM",
    "mamba", "LSTM", "speculative decoding", "KV cache",
    "mixture of experts", "MoE", "flash attention",
    "hardware acceleration", "LLM inference", "scaling",
    "RLHF", "reinforcement learning", "tokenization",
    "neural network", "deep learning", "language model",
]

SOURCES_TO_WATCH = {
    "arxiv_categories": ["cs.LG", "cs.CL", "cs.AI"],
    "github_repos": [
        "karpathy/nanoGPT", "karpathy/micrograd",
        "karpathy/minbpe", "karpathy/llm.c",
        "huggingface/transformers", "state-spaces/mamba",
    ],
    "blogs": [
        "https://karpathy.github.io/",
        "https://lilianweng.github.io/",
        "https://huggingface.co/blog/",
        "https://distill.pub/",
    ],
    "youtube_channels": [
        "UCbfYPyITQ-7l4upoX8nvctg",  # Andrej Karpathy
    ],
}

# ── Client (Vertex AI) ─────────────────────────────────────────────────────
_PROJECT = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "ceo-proj-foxbrain-prod")
_REGION  = os.environ.get("VERTEX_REGION_CLAUDE_4_6_SONNET", "europe-west1")
_MODEL   = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet-4-6")

client = anthropic.AnthropicVertex(project_id=_PROJECT, region=_REGION)


# ── Tool implementations ───────────────────────────────────────────────────

def search_arxiv(query: str, days_back: int = 1, max_results: int = 10) -> list:
    """Search arXiv for recent papers sorted by submission date."""
    base = "http://export.arxiv.org/api/query?"
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    try:
        with urllib.request.urlopen(base + params, timeout=15) as r:
            tree = ET.parse(r)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        cutoff = datetime.now() - timedelta(days=days_back)
        papers = []
        for entry in tree.findall("atom:entry", ns):
            published = entry.find("atom:published", ns).text
            pub_date  = datetime.fromisoformat(published.replace("Z", ""))
            if pub_date < cutoff and days_back <= 7:
                continue
            arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
            papers.append({
                "id":        arxiv_id,
                "title":     entry.find("atom:title",   ns).text.strip(),
                "abstract":  entry.find("atom:summary", ns).text.strip(),
                "published": published,
                "url":       f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url":   f"https://arxiv.org/pdf/{arxiv_id}",
            })
        return papers
    except Exception as e:
        return [{"error": str(e)}]


def get_semantic_scholar_citations(arxiv_id: str, min_citations: int = 500) -> list:
    """Return highly-cited papers that cite a given arXiv paper."""
    try:
        url = (
            f"https://api.semanticscholar.org/graph/v1/"
            f"paper/ArXiv:{arxiv_id}/citations?"
            f"fields=title,citationCount,year,externalIds&limit=50"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "LLMWikiAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        high_impact = []
        for item in data.get("data", []):
            paper = item.get("citing_paper", {})
            if paper.get("citationCount", 0) >= min_citations:
                ext_ids = paper.get("externalIds", {})
                high_impact.append({
                    "title":    paper.get("title", ""),
                    "citations": paper.get("citationCount", 0),
                    "year":     paper.get("year", ""),
                    "arxiv_id": ext_ids.get("ArXiv", ""),
                })
        return sorted(high_impact, key=lambda x: x["citations"], reverse=True)[:10]
    except Exception as e:
        return [{"error": str(e)}]


def fetch_github_readme(repo: str) -> dict:
    """Fetch the raw README of a GitHub repository."""
    try:
        url = f"https://api.github.com/repos/{repo}/readme"
        req = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github.v3.raw",
            "User-Agent": "LLMWikiAgent/1.0",
        })
        with urllib.request.urlopen(req, timeout=15) as r:
            content = r.read().decode("utf-8")
        return {"repo": repo, "content": content[:8000], "url": f"https://github.com/{repo}"}
    except Exception as e:
        return {"error": str(e), "repo": repo}


def fetch_blog_post(url: str) -> dict:
    """Fetch and strip HTML from a blog URL."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LLMWikiAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return {"url": url, "content": text[:6000]}
    except Exception as e:
        return {"error": str(e), "url": url}


def get_youtube_transcript(video_id: str) -> dict:
    """Get auto-generated transcript for a YouTube video via yt-dlp."""
    try:
        import subprocess, tempfile, glob
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    "yt-dlp", "--skip-download",
                    "--write-auto-sub", "--sub-format", "vtt",
                    "--sub-lang", "en",
                    "-o", f"{tmp}/%(title)s",
                    f"https://youtube.com/watch?v={video_id}",
                ],
                capture_output=True, text=True, timeout=60,
            )
            vtt_files = glob.glob(f"{tmp}/*.vtt")
            if vtt_files:
                text = open(vtt_files[0]).read()
                text = re.sub(r"<[^>]+>", "", text)
                text = re.sub(r"\d{2}:\d{2}.*\n", "", text)
                text = re.sub(r"\s+", " ", text).strip()
                return {"video_id": video_id, "transcript": text[:8000]}
    except Exception as e:
        return {"error": str(e), "video_id": video_id}
    return {"error": "transcript unavailable", "video_id": video_id}


def download_pdf(arxiv_id: str, filename: str) -> dict:
    """Download a PDF from arXiv into raw/."""
    out_path = RAW / filename
    if out_path.exists():
        return {"status": "already_exists", "path": str(out_path)}
    try:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        req = urllib.request.Request(pdf_url, headers={"User-Agent": "LLMWikiAgent/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            out_path.write_bytes(r.read())
        return {"status": "downloaded", "path": str(out_path), "filename": filename}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_already_in_wiki(title: str) -> bool:
    """True if a note with a similar title already exists in wiki/."""
    needle = title.lower().replace(" ", "")
    for f in WIKI.glob("*.md"):
        if needle in f.stem.lower().replace(" ", ""):
            return True
    return False


def write_content_note(filename: str, content: str) -> dict:
    """Write a markdown note to wiki/ for non-PDF content (GitHub, blog, YouTube)."""
    out_path = WIKI / filename
    if out_path.exists():
        return {"status": "already_exists", "path": str(out_path)}
    out_path.write_text(content, encoding="utf-8")
    return {"status": "written", "path": str(out_path)}


def log_agent_run(results: dict):
    """Append a run record to agent_log.json."""
    log = []
    if LOG.exists():
        try:
            log = json.loads(LOG.read_text())
        except Exception:
            pass
    log.append({"timestamp": datetime.now().isoformat(), **results})
    LOG.write_text(json.dumps(log, indent=2))


# ── Tool schemas for Claude ────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_arxiv",
        "description": "Search arXiv for recent papers by keyword or topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":       {"type": "string", "description": "Search query"},
                "days_back":   {"type": "integer", "default": 1,  "description": "How many days back to include"},
                "max_results": {"type": "integer", "default": 10, "description": "Max results to return"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_citations",
        "description": "Get high-impact papers (≥ min_citations) that cite an existing wiki paper.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arxiv_id":      {"type": "string",  "description": "arXiv ID of the source paper"},
                "min_citations": {"type": "integer", "default": 500, "description": "Minimum citation count"},
            },
            "required": ["arxiv_id"],
        },
    },
    {
        "name": "fetch_github_readme",
        "description": "Fetch README text from a GitHub repository (owner/repo).",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "GitHub repo in owner/repo format"},
            },
            "required": ["repo"],
        },
    },
    {
        "name": "fetch_blog_post",
        "description": "Fetch and extract text from a blog post URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL of the blog post"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "get_youtube_transcript",
        "description": "Get auto-generated transcript of a YouTube video.",
        "input_schema": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string", "description": "YouTube video ID (11-char string)"},
            },
            "required": ["video_id"],
        },
    },
    {
        "name": "download_pdf",
        "description": "Download an arXiv paper PDF into raw/ for later processing by build_wiki.py.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string", "description": "arXiv ID, e.g. 2312.00752"},
                "filename":  {"type": "string", "description": "Output filename, e.g. Mamba.pdf"},
            },
            "required": ["arxiv_id", "filename"],
        },
    },
    {
        "name": "check_wiki",
        "description": "Check if content with a given title already exists in the wiki.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title to check"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "write_note",
        "description": (
            "Write a markdown note directly to wiki/ for non-PDF content "
            "(GitHub repos, blog posts, YouTube transcripts). "
            "Include full YAML frontmatter: title, tags, year, tldr, wikilinks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Output filename, e.g. nanoGPT.md"},
                "content":  {"type": "string", "description": "Full markdown content including YAML frontmatter"},
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "done",
        "description": "Signal the agent is finished. Provide a summary plus lists of added and skipped items.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "added":   {"type": "array", "items": {"type": "string"}},
                "skipped": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "added", "skipped"],
        },
    },
]


# ── Tool dispatcher ────────────────────────────────────────────────────────

def execute_tool(name: str, inputs: dict) -> str:
    if name == "search_arxiv":
        result = search_arxiv(**inputs)
    elif name == "get_citations":
        result = get_semantic_scholar_citations(
            inputs["arxiv_id"], inputs.get("min_citations", 500)
        )
    elif name == "fetch_github_readme":
        result = fetch_github_readme(inputs["repo"])
    elif name == "fetch_blog_post":
        result = fetch_blog_post(inputs["url"])
    elif name == "get_youtube_transcript":
        result = get_youtube_transcript(inputs["video_id"])
    elif name == "download_pdf":
        result = download_pdf(inputs["arxiv_id"], inputs["filename"])
    elif name == "check_wiki":
        result = {"exists": check_already_in_wiki(inputs["title"])}
    elif name == "write_note":
        result = write_content_note(inputs["filename"], inputs["content"])
    elif name == "done":
        result = inputs
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result)


# ── System prompts ─────────────────────────────────────────────────────────

def _base_context(existing: list[str]) -> str:
    return (
        f"Wiki themes: {', '.join(WIKI_THEMES)}.\n"
        f"Already in wiki (skip duplicates): {', '.join(existing[:30])}.\n\n"
        "For each candidate:\n"
        "1. call check_wiki first to avoid duplicates\n"
        "2. score relevance 1–10; only proceed if ≥ 7\n"
        "3. for arXiv papers → download_pdf\n"
        "4. for GitHub/blog/YouTube → write_note with full YAML frontmatter "
        "(title, authors, year, tags, tldr) and [[wikilinks]] to existing notes\n"
        "5. call done when finished with summary, added list, skipped list"
    )


def _make_system(mode: str, extra: str, existing: list[str]) -> list[dict]:
    text = (
        f"You are a research ingestion agent for an AI/ML knowledge wiki. "
        f"Mode: {mode}.\n\n"
        + extra
        + "\n\n"
        + _base_context(existing)
    )
    # Cache the (large, stable) system prompt
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


# ── Main agent loop ────────────────────────────────────────────────────────

def run_agent(mode: str = "daily", topic: str = None) -> dict:
    existing = [f.stem for f in WIKI.glob("*.md")]

    if mode == "topic" and topic:
        extra = f'The user wants to add content about: "{topic}". Search for the most relevant and high-quality papers, repos, or blog posts on this topic.'
        messages = [{"role": "user", "content": f"Add content about: {topic}"}]

    elif mode == "citations":
        wiki_arxiv_ids = {
            "2312.00752": "Mamba",
            "2405.21060": "Transformers are SSMs (Mamba-2)",
            "2405.04517": "xLSTM",
            "2205.14135": "FlashAttention",
            "2307.08691": "FlashAttention-2",
            "2111.00396": "S4",
            "2305.13048": "RWKV",
            "2307.08621": "RetNet",
        }
        extra = (
            "Check each wiki paper for highly-cited new papers that cite it "
            "(min 500 citations). For each high-impact citing paper NOT already in the wiki, download it."
        )
        messages = [{"role": "user", "content": f"Check citations for: {json.dumps(wiki_arxiv_ids)}"}]

    else:  # daily
        extra = (
            f"Search arXiv for papers from the last 24 hours in: {', '.join(WIKI_THEMES[:8])}. "
            f"Also check these GitHub repos for notable updates: {', '.join(SOURCES_TO_WATCH['github_repos'][:4])}."
        )
        messages = [{"role": "user", "content": "Run daily monitoring check."}]

    system = _make_system(mode, extra, existing)

    added   = []
    skipped = []

    for iteration in range(20):
        response = client.messages.create(
            model=_MODEL,
            max_tokens=4096,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        tool_results = []
        finished     = False

        for block in response.content:
            if block.type != "tool_use":
                continue

            print(f"  → {block.name}({json.dumps(block.input)[:100]})")
            result = execute_tool(block.name, block.input)

            if block.name == "done":
                data    = json.loads(result)
                added   = data.get("added", [])
                skipped = data.get("skipped", [])
                finished = True
            elif block.name == "download_pdf":
                r = json.loads(result)
                if r.get("status") == "downloaded":
                    added.append(r["filename"])
            elif block.name == "write_note":
                r = json.loads(result)
                if r.get("status") == "written":
                    added.append(r["path"])

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        if finished:
            break

        messages.append({"role": "user", "content": tool_results})

    return {
        "mode":    mode,
        "topic":   topic,
        "added":   added,
        "skipped": skipped,
        "timestamp": datetime.now().isoformat(),
    }


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode  = sys.argv[1] if len(sys.argv) > 1 else "daily"
    topic = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None

    print(f"\nRunning agent — mode: '{mode}'" + (f", topic: '{topic}'" if topic else ""))
    print("-" * 60)

    results = run_agent(mode=mode, topic=topic)
    log_agent_run(results)

    print(f"\nAdded  : {len(results['added'])} item(s)")
    for item in results["added"]:
        print(f"  + {item}")
    print(f"Skipped: {len(results['skipped'])} item(s)")
    for item in results["skipped"]:
        print(f"  - {item}")

    if results["added"]:
        print("\nRebuilding wiki and pushing…")
        import subprocess
        subprocess.run(["python3", "build_wiki.py"], cwd=VAULT)
        subprocess.run(["python3", "build.py"],      cwd=VAULT)
        subprocess.run(["git", "add", "."],          cwd=VAULT)
        subprocess.run(
            ["git", "commit", "-m",
             f"Agent ({mode}): added {len(results['added'])} item(s)"],
            cwd=VAULT,
        )
        subprocess.run(["git", "push", "origin", "main"], cwd=VAULT)
        print("Done — wiki updated and pushed.")
