#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
education.py — Educational content generator from Knowledge Bases *and* external sources.

- Mines topics automatically from provided sources (URLs, sites.yaml, feeds.yaml) or uses explicit --topic/--topics-file.
- Generates per-topic artifacts: tutorial.md, methodology.md, checklist.md, slides.md, linkedin.txt
- Stashes scraped pages to kb_stash/ so they become part of your KB.
- Uses Ollama-compatible LLM calls.

Persistent:
  output/registry.json (dedupe topics), output/index.json (KB debug)

Environment (.env):
  OLLAMA_BASE=http://localhost:11434
  OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
  OLLAMA_TEMPERATURE=0.3
  PEXELS_API_KEY=...     # optional
  UNSPLASH_ACCESS_KEY=...# optional

Requirements:
  pip install beautifulsoup4 python-dotenv pyyaml requests pdfminer.six feedparser
"""

from __future__ import annotations

import argparse
import datetime as dt
from datetime import timezone
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Optional PDF (support both import paths)
try:
    from pdfminer_high_level import extract_text as _pdf_extract_text  # type: ignore
    pdf_extract_text = _pdf_extract_text
except Exception:
    try:
        from pdfminer.high_level import extract_text as _pdf_extract_text2  # type: ignore
        pdf_extract_text = _pdf_extract_text2
    except Exception:
        pdf_extract_text = None

# ======================
# Env
# ======================
load_dotenv()

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

UA_HDRS = {"User-Agent": "Mozilla/5.0 (compatible; DFieldKB/1.1)"}

TEXT_EXT = {".md", ".txt"}
HTML_EXT = {".html", ".htm"}
PDF_EXT  = {".pdf"}

# ======================
# Utilities
# ======================

def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)

def now_iso() -> str:
    return dt.datetime.now(timezone.utc).isoformat()

def make_topic_id(topic: str) -> str:
    s = slugify(topic)[:70]
    h = hashlib.sha1(topic.strip().lower().encode("utf-8")).hexdigest()[:8]
    date = dt.datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{date}-{s}-{h}"

def to_abs(base: str, href: str) -> str:
    try:
        return urljoin(base, href)
    except Exception:
        return href

def safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return path.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return ""

# ======================
# KB loading & chunking
# ======================

@dataclass
class KBChunk:
    file_path: str
    title: str
    section: str
    text: str

@dataclass
class KBFile:
    path: str
    title: str
    chunks: List[KBChunk]

def read_text_file(path: Path) -> str:
    return safe_read(path)

def read_html_file(path: Path) -> str:
    txt = safe_read(path)
    soup = BeautifulSoup(txt, "html.parser")
    for t in soup(["script", "style", "noscript"]): t.decompose()
    return soup.get_text(" ", strip=True)

def read_pdf_file(path: Path) -> str:
    if not pdf_extract_text:
        print(f"⚠️  pdfminer.six not installed, skipping PDF: {path.name}")
        return ""
    try:
        return pdf_extract_text(str(path))
    except Exception as e:
        print(f"⚠️  PDF read failed: {path.name} ({e})")
        return ""

def split_into_chunks(raw_text: str, file_title: str, file_path: str, max_chars: int = 1400) -> List[KBChunk]:
    if not raw_text.strip():
        return []
    parts = re.split(r"\n(?=#+\s)|\n{2,}", raw_text)
    chunks: List[KBChunk] = []
    buf, section = [], file_title or Path(file_path).stem
    for p in parts:
        p = p.strip()
        if not p: continue
        if p.startswith("#"):
            if buf:
                text = " ".join(buf).strip()
                chunks.extend(_cap_chunk(text, file_path, file_title, section, max_chars))
                buf = []
            section = p.splitlines()[0].lstrip("# ").strip()[:120] or section
        else:
            buf.append(p)
    if buf:
        text = " ".join(buf).strip()
        chunks.extend(_cap_chunk(text, file_path, file_title, section, max_chars))
    return chunks

def _cap_chunk(text: str, file_path: str, file_title: str, section: str, max_chars: int) -> List[KBChunk]:
    if len(text) <= max_chars:
        return [KBChunk(file_path, file_title, section, text)]
    out = []
    paragraphs = re.split(r"\n{2,}|(?<=\.)\s{2,}", text)
    cur = []
    for para in paragraphs:
        if sum(len(x) for x in cur) + len(para) + 1 <= max_chars:
            cur.append(para)
        else:
            out.append(KBChunk(file_path, file_title, section, " ".join(cur)))
            cur = [para]
    if cur:
        out.append(KBChunk(file_path, file_title, section, " ".join(cur)))
    return out

def gather_paths(kb_dirs: List[Path], kb_files: List[Path]) -> List[Path]:
    paths: List[Path] = []
    for d in kb_dirs:
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in (TEXT_EXT | HTML_EXT | PDF_EXT):
                paths.append(p)
    for f in kb_files:
        if f.is_file():
            paths.append(f)
    # de-dup
    seen, unique = set(), []
    for p in paths:
        sp = str(p.resolve())
        if sp not in seen:
            seen.add(sp); unique.append(p)
    return unique

def load_kb_from_paths(paths: List[Path]) -> List[KBFile]:
    kb: List[KBFile] = []
    for p in paths:
        ext = p.suffix.lower()
        if ext in TEXT_EXT:
            raw = read_text_file(p)
        elif ext in HTML_EXT:
            raw = read_html_file(p)
        elif ext in PDF_EXT:
            raw = read_pdf_file(p)
        else:
            continue
        title = p.stem.replace("_"," ").replace("-"," ").title()
        chunks = split_into_chunks(raw, title, str(p))
        if chunks:
            kb.append(KBFile(str(p), title, chunks))
    return kb

# ======================
# Simple ranking
# ======================

def score_chunk(chunk: KBChunk, query_terms: List[str]) -> float:
    text = chunk.text.lower()
    score = 0.0
    for t in query_terms:
        if not t: continue
        k = t.lower()
        score += text.count(k) * 3.0
    score += min(len(chunk.text) / 800.0, 1.0)
    return score

def search_kb(kb: List[KBFile], topic: str, k: int = 10) -> Tuple[List[KBChunk], List[str]]:
    terms = re.split(r"[\s,;/]+", topic.strip())
    all_chunks: List[KBChunk] = [c for f in kb for c in f.chunks]
    ranked = sorted(all_chunks, key=lambda c: score_chunk(c, terms), reverse=True)
    top = [c for c in ranked[:k] if len(c.text) > 200]
    sources = compact_sources(top, max_sources=8)
    return top, sources

def compact_sources(chunks: List[KBChunk], max_sources: int = 8) -> List[str]:
    out, seen = [], set()
    for c in chunks:
        key = (Path(c.file_path).name, c.section[:80])
        if key in seen: continue
        seen.add(key)
        out.append(f"{Path(c.file_path).name} :: {c.section}")
        if len(out) >= max_sources: break
    return out

# ======================
# External Source Fetching (URLs, sites.yaml, feeds.yaml)
# ======================

def fetch_html(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=UA_HDRS, timeout=30)
        if r.status_code >= 400:
            return None
        return r.text
    except Exception:
        return None

def scrape_page_to_markdown(url: str) -> Optional[str]:
    doc = fetch_html(url)
    if not doc: return None
    soup = BeautifulSoup(doc, "html.parser")
    for t in soup(["script","style","noscript","svg"]): t.decompose()
    title = extract_title(soup) or url
    body = extract_main_text(soup)
    md = f"# {title}\n\n{body}\n"
    return md

def extract_title(soup: BeautifulSoup) -> str:
    for sel in ['meta[property="og:title"]','meta[name="twitter:title"]']:
        tag = soup.select_one(sel)
        if tag and tag.get("content"): return tag["content"].strip()
    if soup.title and soup.title.text: return soup.title.text.strip()
    h1 = soup.find("h1")
    return h1.get_text(" ", strip=True) if h1 else ""

def extract_main_text(soup: BeautifulSoup) -> str:
    candidates = []
    for sel in ["article","main","#content",".content",".post",".article__body",".entry-content"]:
        candidates += soup.select(sel)
    if not candidates: candidates = [soup.body or soup]
    best_text, best_len = "", 0
    for node in candidates:
        parts = []
        for p in node.find_all("p"):
            t = p.get_text(" ", strip=True)
            if len(t) >= 40: parts.append(t)
        text = "\n\n".join(parts)
        if len(text) > best_len:
            best_text, best_len = text, len(text)
    if best_len < 200:
        parts = []
        for p in soup.find_all("p"):
            t = p.get_text(" ", strip=True)
            if len(t) >= 40: parts.append(t)
        best_text = "\n\n".join(parts)
    return best_text.strip()

def crawl_site(start_url: str, link_selector: Optional[str], limit: int) -> List[str]:
    doc = fetch_html(start_url)
    if not doc: return []
    soup = BeautifulSoup(doc, "html.parser")
    links = set()
    if link_selector:
        for a in soup.select(link_selector)[:limit*4]:
            href = a.get("href")
            if href: links.add(to_abs(start_url, href))
    else:
        for a in soup.find_all("a", href=True)[:limit*10]:
            href = a["href"]
            abs_url = to_abs(start_url, href)
            if re.search(r"/20\d{2}/|/news/|/story/|/article/", abs_url) or any(k in abs_url.lower() for k in ["ai","ml","block","crypto","cyber","security","nis2","mlops"]):
                links.add(abs_url)
    base_host = urlparse(start_url).netloc.split(":")[0]
    links = [u for u in links if urlparse(u).netloc.endswith(base_host)]
    return links[:limit]

def fetch_from_sites_yaml(path: str) -> List[Tuple[str,str]]:
    """Return list of (url, topic_hint)."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out: List[Tuple[str,str]] = []
    for s in cfg.get("sites", []):
        start = s.get("start_url"); sel = s.get("link_selector"); lim = int(s.get("limit", 5))
        hint = s.get("topic") or s.get("topic_hint") or ""
        if not start: continue
        for u in crawl_site(start, sel, lim):
            out.append((u, hint))
    return out

def fetch_from_feeds_yaml(path: str, limit_per_topic: int = 6) -> List[Tuple[str, str]]:
    """Return list of (url, topic_hint) from RSS.

    Accepts two formats under top-level `topics`:
      topics:
        ai:
          - https://feed1.xml
          - https://feed2.xml
        cybersecurity:
          rss:
            - https://feed3.xml
            - https://feed4.xml
    """
    import feedparser
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out: List[Tuple[str, str]] = []

    topics_node = cfg.get("topics", {})
    if not isinstance(topics_node, dict):
        return out

    for topic, node in topics_node.items():
        feeds: List[str] = []
        if isinstance(node, list):
            feeds = [u for u in node if isinstance(u, str) and u.strip()]
        elif isinstance(node, dict):
            if "rss" in node and isinstance(node["rss"], list):
                feeds = [u for u in node["rss"] if isinstance(u, str) and u.strip()]
            else:
                for _, v in node.items():
                    if isinstance(v, list):
                        feeds.extend([u for u in v if isinstance(u, str) and u.strip()])

        for rss in feeds:
            try:
                feed = feedparser.parse(rss)
                for e in feed.entries[:limit_per_topic]:
                    link = getattr(e, "link", "")
                    if link:
                        out.append((link, topic))
            except Exception:
                continue

    return out

# ======================
# Topic Mining
# ======================

def mine_topics_from_markdown(md_docs: List[Tuple[str,str]], max_topics: int = 12) -> List[str]:
    """
    md_docs: list of (source_name, markdown_text)
    Heuristics: H1/H2/H3 titles + frequent noun phrases.
    """
    candidates: List[str] = []
    # 1) Headings
    for _, md in md_docs:
        for m in re.finditer(r"^\s{0,3}(#{1,3})\s+(.+)$", md, flags=re.M):
            title = m.group(2).strip()
            if 8 <= len(title) <= 90:
                candidates.append(title)
    # 2) Simple noun phrase extraction (very light)
    for _, md in md_docs:
        text = re.sub(r"[^\w\s-]", " ", md)
        words = [w for w in text.split() if len(w) > 2]
        for i in range(len(words)-2):
            tri = " ".join(words[i:i+3])
            if re.search(r"\b(ai|ml|model|risk|policy|governance|pipeline|workflow|token|stablecoin|wallet|zero|trust|identity|access|control|incident|reporting|detection|response|nis2|gdpr)\b", tri.lower()):
                candidates.append(tri.title())
    # normalize/dedup
    clean = []
    seen = set()
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip(" -—–")
        if len(c) < 8 or len(c) > 90: continue
        key = c.lower()
        if key in seen: continue
        seen.add(key); clean.append(c)
    # simple sort by “interestingness”
    def interest(s: str) -> float:
        base = min(len(s)/60.0, 1.0)
        kw = 1.0 if re.search(r"(ai|ml|block|crypto|cyber|security|nis2|gdpr|zero trust|governance|lifecycle|checklist|methodology)", s.lower()) else 0.0
        return base + kw
    clean.sort(key=interest, reverse=True)
    return clean[:max_topics]

def refine_topics_with_llm(candidates: List[str], max_topics: int, language: str) -> List[str]:
    if not candidates: return []
    prompt = f"""You are selecting practical training topics for working professionals.

Input list:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Task:
- Merge near-duplicates.
- Keep the {max_topics} most useful, specific, teachable topics.
- Return ONLY a JSON array of strings in {language} with no commentary."""
    refined = llm(prompt)
    try:
        arr = json.loads(refined.strip().splitlines()[-1])
        arr = [s for s in arr if isinstance(s, str)]
        out = []
        seen = set()
        for s in arr:
            s = s.strip()
            if 8 <= len(s) <= 90 and s.lower() not in seen:
                seen.add(s.lower()); out.append(s)
        return out[:max_topics]
    except Exception:
        return candidates[:max_topics]


# ======================
# Prompts (education)
# ======================

BASE_SYSTEM = """You are a senior instructor and technical writer.

Follow strictly:
1) Use ONLY the provided context. If something is missing, write: "Not in provided sources."
2) Be precise and pragmatic. Prefer standards, dates, and vendor-neutral guidance.
3) Audience: experienced practitioners. Assume familiarity with fundamentals.
4) Short sentences. No filler. No hype. No self-references.
5) Lists must use "- " bullets. Avoid numbered lists unless steps truly must be ordered.
6) Prefer concrete commands, configs, code, and diagnostics over generic advice.
"""

COMMON_STYLE = """Style guardrails:

- Start with the core idea in plain language. Then deepen with specifics.
- Use "- " bullets for concise points. Keep bullets tight (one idea each).
- Include copy-pasteable snippets (CLI, config, code) where helpful.
- Call out failure modes and verification steps.
- End with a brief "Next steps" section for individual contributors and for teams.
- Never write "In this article" or similar blog tropes.
"""

TUTORIAL_TMPL = f"""{COMMON_STYLE}
Write a tutorial in {{language}} based ONLY on the context.

Topic:
{{topic}}

Context (excerpts):
---
{{context}}
---

Output (Markdown with these H2 headings):

## Overview

## Key concepts

## Hands-on: step by step
- prerequisites
- setup
- run/execute
- validate
- troubleshoot

## Examples
- minimal example
- realistic example

## Common pitfalls

## Next steps (ICs vs teams)

## Sources used
"""

METHODOLOGY_TMPL = f"""{COMMON_STYLE}
Write an operating procedure in {{language}} based ONLY on the context.

Topic:
{{topic}}

Context:
---
{{context}}
---

Output (Markdown):

## Purpose

## When to use

## Prerequisites

## Procedure
- prepare
- execute
- verify
- rollback

## Definition of done

## Governance & checks

## Sources used
"""

CHECKLIST_TMPL = f"""{COMMON_STYLE}
Create a checklist in {{language}} based ONLY on the context.

Topic:
{{topic}}

Context:
---
{{context}}
---

Output (Markdown):

## Checklist
- [ ] 10–18 actionable items focused on doing, not describing

## Notes

## Sources used
"""

SLIDES_TMPL = f"""{COMMON_STYLE}
Create a slide outline in {{language}} based ONLY on the context.

Topic:
{{topic}}

Context:
---
{{context}}
---

Output (Markdown):

# Title slide

## Agenda

## Slide 1: Problem
- what breaks
- why it matters
- baseline

## Slide 2: Concepts
- terms
- mental model
- boundaries

## Slide 3: Setup
- prereqs
- config
- gotchas

## Slide 4: Workflow
- steps
- checkpoints
- artifacts

## Slide 5: Example
- input
- output
- validation

## Slide 6: Failure modes
- symptom
- cause
- fix

## Slide 7: Observability
- metrics
- logs
- tests

## Slide 8: Hardening
- constraints
- limits
- controls

## Slide 9: Next steps
- for ICs
- for teams
- for leads

## Slide 10: References

## Sources used
"""

LI_TMPL = f"""{COMMON_STYLE}
Write an educational LinkedIn post in {{language}} based ONLY on the context (900–1400 chars).
No URLs. No hashtags.

Topic:
{{topic}}

Context:
---
{{context}}
---

Output: plain text only.
"""

# ======================
# Ollama
# ======================

def _ollama_openai_compat(prompt: str) -> Optional[str]:
    try:
        url = f"{OLLAMA_BASE}/v1/chat/completions"
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": BASE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": OLLAMA_TEMPERATURE,
        }
        r = requests.post(url, json=payload, timeout=180)
        if r.status_code >= 400:
            return None
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None

def _ollama_native_chat(prompt: str) -> Optional[str]:
    try:
        url = f"{OLLAMA_BASE}/api/chat"
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": BASE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": OLLAMA_TEMPERATURE},
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=180)
        if r.status_code >= 400:
            return None
        return r.json()["message"]["content"]
    except Exception:
        return None

def _ollama_native_generate(prompt: str) -> Optional[str]:
    try:
        url = f"{OLLAMA_BASE}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"System:\n{BASE_SYSTEM}\n\nUser:\n{prompt}",
            "options": {"temperature": OLLAMA_TEMPERATURE},
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=180)
        if r.status_code >= 400:
            return None
        return r.json()["response"]
    except Exception:
        return None

def _ollama_cli(prompt: str) -> str:
    if not shutil.which("ollama"):
        raise RuntimeError("Ollama CLI not found and HTTP API failed.")
    proc = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=f"System:\n{BASE_SYSTEM}\n\nUser:\n{prompt}".encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "ignore"))
    return proc.stdout.decode("utf-8", "ignore")

def llm(prompt: str) -> Optional[str]:
    return (
        _ollama_openai_compat(prompt)
        or _ollama_native_chat(prompt)
        or _ollama_native_generate(prompt)
        or _ollama_cli(prompt)
    )

def llm_safe(prompt: str, fallback: str = "Not in provided sources.") -> str:
    try:
        out = llm(prompt)
        if not out or not isinstance(out, str):
            return fallback
        return out
    except Exception:
        return fallback

# ======================
# Images (optional)
# ======================

def _search_pexels(q: str, per_page: int = 2) -> List[Dict[str, Any]]:
    if not PEXELS_API_KEY: return []
    try:
        r = requests.get("https://api.pexels.com/v1/search",
                         headers={"Authorization": PEXELS_API_KEY},
                         params={"query": q, "per_page": per_page},
                         timeout=30)
        r.raise_for_status()
        out = []
        for p in r.json().get("photos", []):
            out.append({"url": p["src"]["large"], "photographer": p.get("photographer"),
                        "credit_url": p.get("url"), "provider": "Pexels"})
        return out
    except Exception:
        return []

def _search_unsplash(q: str, per_page: int = 2) -> List[Dict[str, Any]]:
    if not UNSPLASH_ACCESS_KEY: return []
    try:
        r = requests.get("https://api.unsplash.com/search/photos",
                         params={"query": q, "per_page": per_page, "client_id": UNSPLASH_ACCESS_KEY},
                         timeout=30)
        r.raise_for_status()
        out = []
        for p in r.json().get("results", []):
            link = p.get("urls", {}).get("regular") or p.get("urls", {}).get("full")
            credit = p.get("user", {}).get("links", {}).get("html")
            out.append({"url": link, "photographer": p.get("user", {}).get("name"),
                        "credit_url": credit, "provider": "Unsplash"})
        return out
    except Exception:
        return []

def choose_images(topic: str, count: int = 2) -> List[Dict[str, Any]]:
    queries = [topic, f"{topic} EU", f"{topic} diagram", f"{topic} Hungary"]
    out: List[Dict[str, Any]] = []
    for q in queries:
        if len(out) >= count: break
        res = _search_pexels(q, 2) or _search_unsplash(q, 2)
        if res: out.append(res[0])
    return out[:count]

def download_images(imgs: List[Dict[str, Any]], dest_dir: Path) -> List[Dict[str, Any]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, im in enumerate(imgs, start=1):
        path = dest_dir / f"img_{i}.jpg"
        try:
            r = requests.get(im["url"], headers=UA_HDRS, timeout=60)
            r.raise_for_status()
            path.write_bytes(r.content)
            meta = dict(im); meta["local_path"] = str(Path("images") / path.name)
            saved.append(meta)
            print(f"🖼️  saved: {path.name}")
        except Exception as e:
            print(f"⚠️  image download failed: {e}")
    return saved

# ======================
# Registry (dedupe topics)
# ======================

@dataclass
class TopicItem:
    topic: str
    topic_id: str
    created_at: str
    language: str

class TopicRegistry:
    def __init__(self, path: Path, dedupe_days: int = 120):
        self.path = path
        self.dedupe_days = dedupe_days
        # always start with the full schema
        self.data: Dict[str, Any] = {"seen": {}, "items": [], "tally": {}}
        self._load()

    def _normalize(self):
        # repair missing keys or wrong types
        if not isinstance(self.data, dict):
            self.data = {"seen": {}, "items": [], "tally": {}}
            return
        self.data.setdefault("seen", {})
        self.data.setdefault("items", [])
        self.data.setdefault("tally", {})
        if not isinstance(self.data["seen"], dict): self.data["seen"] = {}
        if not isinstance(self.data["items"], list): self.data["items"] = []
        if not isinstance(self.data["tally"], dict): self.data["tally"] = {}

    def _load(self):
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                self.data = loaded
            except Exception:
                # backup the corrupt file and start fresh
                try:
                    self.path.replace(self.path.with_suffix(".bak"))
                except Exception:
                    pass
                self.data = {"seen": {}, "items": [], "tally": {}}
        self._normalize()

    def save(self):
        self._normalize()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _fresh(self, iso: str) -> bool:
        if self.dedupe_days <= 0: return True
        try:
            then = dt.datetime.fromisoformat(iso.replace("Z","")).replace(tzinfo=timezone.utc)
            return then >= (dt.datetime.now(timezone.utc) - dt.timedelta(days=self.dedupe_days))
        except Exception:
            return True

    def key(self, topic: str, language: str) -> str:
        return hashlib.sha1(f"{topic.strip().lower()}|{language}".encode("utf-8")).hexdigest()

    def is_done(self, topic: str, language: str) -> bool:
        self._normalize()
        k = self.key(topic, language)
        iso = self.data["seen"].get(k)
        return bool(iso and self._fresh(iso))

    def record(self, topic: str, topic_id: str, language: str):
        self._normalize()
        k = self.key(topic, language)
        now = now_iso()
        self.data["seen"][k] = now
        self.data["items"].append(asdict(TopicItem(topic=topic, topic_id=topic_id, created_at=now, language=language)))
        # robust tally update even if keys are missing/corrupt
        tkey = f"{language}:{topic.split()[0].lower()}"
        self.data.setdefault("tally", {})
        self.data["tally"][tkey] = int(self.data["tally"].get(tkey, 0)) + 1

# ======================
# Rendering
# ======================

def fm_date() -> str:
    return dt.datetime.now(timezone.utc).isoformat()

def render_with_front_matter(title: str, language: str, topic: str, sources: List[str],
                             hero: Optional[Dict[str, Any]], body_md: str) -> str:
    fm_obj = {
        "title": title,
        "slug": slugify(title),
        "date": fm_date(),
        "language": language,
        "topic": topic,
        "hero_image": hero.get("local_path") if hero else None,
        "sources_used": sources,
    }
    fm = "---\n" + yaml.safe_dump(fm_obj, sort_keys=False, allow_unicode=True) + "---\n\n"
    hero_block = ""
    if hero:
        hero_block = f'![{html.escape(title)}]({hero["local_path"]})\n*Photo: {hero.get("photographer","Unknown")} via {hero.get("provider")}*\n\n'
    return fm + hero_block + body_md.strip() + "\n"

# ======================
# Generation per topic
# ======================

def clamp_chars(s: str, limit: int) -> str:
    return s if len(s) <= limit else (s[:limit] + "\n...")

def join_context(chunks: List[KBChunk], max_chars: int = 6000) -> str:
    lines = []
    used = 0
    for c in chunks:
        block = f"[{Path(c.file_path).name} :: {c.section}]\n{c.text.strip()}\n"
        if used + len(block) > max_chars: break
        lines.append(block); used += len(block)
    return "\n".join(lines).strip()

def validate_linkedin(text: str, max_chars: int = 1400) -> str:
    s = re.sub(r"\s+\n", "\n", text.strip())
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s if len(s) <= max_chars else (s[:max_chars-1] + "…")

def generate_topic_artifacts(topic: str, language: str, kb: List[KBFile]) -> Dict[str, Any]:
    chunks, sources = search_kb(kb, topic, k=12)
    if not chunks:
        raise RuntimeError(f"No relevant KB content for topic: {topic}")
    context = join_context(chunks, max_chars=7000)

    tutorial = llm_safe(TUTORIAL_TMPL.format(language=language, topic=topic, context=context))
    methodology = llm_safe(METHODOLOGY_TMPL.format(language=language, topic=topic, context=context))
    checklist = llm_safe(CHECKLIST_TMPL.format(language=language, topic=topic, context=context))
    slides = llm_safe(SLIDES_TMPL.format(language=language, topic=topic, context=context))
    linkedin_raw = llm_safe(LI_TMPL.format(language=language, topic=topic, context=clamp_chars(context, 2500)))

    return {
        "sources": sources,
        "context_excerpt": clamp_chars(context, 1200),
        "tutorial_md": tutorial,
        "methodology_md": methodology,
        "checklist_md": checklist,
        "slides_md": slides,
        "linkedin_txt": validate_linkedin(linkedin_raw, 1400),
    }

def write_topic_bundle(out_base: Path, topic: str, language: str, artifacts: Dict[str, Any],
                       images: List[Dict[str, Any]]) -> Tuple[Path, str]:
    topic_id = make_topic_id(topic)
    out_dir = out_base / topic_id
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    hero = images[0] if images else None
    files = {
        "tutorial.md": render_with_front_matter(f"{topic} — Tutorial", language, topic, artifacts["sources"], hero, artifacts["tutorial_md"]),
        "methodology.md": render_with_front_matter(f"{topic} — Methodology", language, topic, artifacts["sources"], hero, artifacts["methodology_md"]),
        "checklist.md": render_with_front_matter(f"{topic} — Checklist", language, topic, artifacts["sources"], hero, artifacts["checklist_md"]),
        "slides.md": render_with_front_matter(f"{topic} — Slide Outline", language, topic, artifacts["sources"], hero, artifacts["slides_md"]),
        "linkedin.txt": artifacts["linkedin_txt"],
    }
    for name, content in files.items():
        (out_dir / name).write_text(content, encoding="utf-8")

    ready = {
        "topic": topic, "topic_id": topic_id, "language": language,
        "channels": {
            "tutorial_markdown": files["tutorial.md"],
            "methodology_markdown": files["methodology.md"],
            "checklist_markdown": files["checklist.md"],
            "slides_markdown": files["slides.md"],
            "linkedin_text": files["linkedin.txt"],
        },
        "assets": {
            "hero": hero.get("local_path") if hero else None,
            "images": [im["local_path"] for im in images],
            "image_credits": [{"photographer": im.get("photographer"),
                               "provider": im.get("provider"),
                               "credit_url": im.get("credit_url")} for im in images],
        },
        "sources": artifacts["sources"],
        "context_excerpt": artifacts["context_excerpt"],
    }
    (out_dir / "ready.json").write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "topic": topic, "topic_id": topic_id, "language": language,
        "time": now_iso(), "ollama_model": OLLAMA_MODEL,
        "images": images, "sources": artifacts["sources"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir, topic_id

# ======================
# CLI
# ======================

def parse_args():
    ap = argparse.ArgumentParser(description="KB+Sources → Educational content (Ollama)")

    g_kb = ap.add_argument_group("Knowledge base (local files)")
    g_kb.add_argument("--kb-dir", action="append", default=[], help="Directory with *.md, *.txt, *.html, *.pdf (repeatable)")
    g_kb.add_argument("--kb-file", action="append", default=[], help="Single file path (repeatable)")
    g_kb.add_argument("--index-out", default="output/index.json", help="Where to write KB index debug JSON")

    g_src = ap.add_argument_group("External sources (optional)")
    g_src.add_argument("--source-url", action="append", default=[], help="Scrape a specific URL (repeatable)")
    g_src.add_argument("--sites", help="sites.yaml with site start pages & optional link_selector")
    g_src.add_argument("--feeds", help="feeds.yaml with topic→RSS arrays")
    g_src.add_argument("--stash-dir", default="kb_stash", help="Where to save scraped pages as .md for indexing")

    g_topics = ap.add_argument_group("Topics")
    g_topics.add_argument("--topic", action="append", default=[], help="Explicit topic (repeatable)")
    g_topics.add_argument("--topics-file", help="Text file with one topic per line")
    g_topics.add_argument("--auto-topics", type=int, default=0, help="If >0, mine up to N topics from sources")
    g_topics.add_argument("--llm-refine-topics", action="store_true", help="Use LLM to refine mined topics")
    g_topics.add_argument("--language", default="en", choices=["en","hu"])

    g_out = ap.add_argument_group("Output & control")
    g_out.add_argument("--output-dir", default="output_kb")
    g_out.add_argument("--max-topics", type=int, default=6)
    g_out.add_argument("--registry-path", default="output/registry.json")
    g_out.add_argument("--dedupe-days", type=int, default=120)
    g_out.add_argument("--image-count", type=int, default=2)

    return ap.parse_args()

# ======================
# Main
# ======================

def main():
    args = parse_args()

    # 1) Optionally fetch external sources and stash them as Markdown into kb_stash/
    stash_dir = Path(args.stash_dir)
    stashed_docs: List[Tuple[str,str]] = []  # (name, markdown)
    if args.source_url:
        stash_dir.mkdir(parents=True, exist_ok=True)
        for u in args.source_url:
            md = scrape_page_to_markdown(u)
            if not md: continue
            name = slugify(u)[:80] or "page"
            (stash_dir / f"{name}.md").write_text(md, encoding="utf-8")
            stashed_docs.append((name, md))
    if args.sites:
        for u, hint in fetch_from_sites_yaml(args.sites):
            md = scrape_page_to_markdown(u)
            if not md: continue
            name = slugify(u)[:80] or "site"
            stash_dir.mkdir(parents=True, exist_ok=True)
            (stash_dir / f"{name}.md").write_text(md, encoding="utf-8")
            stashed_docs.append((name, md))
    if args.feeds:
        for u, hint in fetch_from_feeds_yaml(args.feeds, limit_per_topic=6):
            md = scrape_page_to_markdown(u)
            if not md: continue
            name = slugify(u)[:80] or "feed"
            stash_dir.mkdir(parents=True, exist_ok=True)
            (stash_dir / f"{name}.md").write_text(md, encoding="utf-8")
            stashed_docs.append((name, md))

    # 2) Gather KB paths: kb dirs/files + stashed pages
    kb_dirs = [Path(p) for p in args.kb_dir]
    kb_files = [Path(p) for p in args.kb_file]
    if stash_dir.exists():
        kb_dirs.append(stash_dir)
    kb_paths = gather_paths(kb_dirs, kb_files)
    if not kb_paths:
        raise SystemExit("No KB files found. Use --kb-dir/--kb-file or provide sources (--source-url/--sites/--feeds).")

    kb = load_kb_from_paths(kb_paths)
    if not kb:
        raise SystemExit("KB could not be loaded (no parsable content).")

    # 3) Write KB index debug
    index_dbg = {"files": [{"path": f.path, "title": f.title, "chunks": len(f.chunks)} for f in kb]}
    Path(args.index_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.index_out).write_text(json.dumps(index_dbg, ensure_ascii=False, indent=2), encoding="utf-8")

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    registry = TopicRegistry(Path(args.registry_path), dedupe_days=args.dedupe_days)

    # 4) Build the topic list:
    topics: List[str] = []
    if args.topic:
        topics += args.topic
    if args.topics_file:
        topics += [line.strip() for line in Path(args.topics_file).read_text(encoding="utf-8").splitlines() if line.strip()]

    if args.auto_topics > 0:
        # Mine from stashed docs (preferred) or from the first few KB files
        if not stashed_docs:
            for f in kb_paths[:10]:
                ext = f.suffix.lower()
                if ext in TEXT_EXT:
                    stashed_docs.append((f.name, read_text_file(f)))
                elif ext in HTML_EXT:
                    stashed_docs.append((f.name, read_html_file(f)))
        mined = mine_topics_from_markdown(stashed_docs, max_topics=args.auto_topics * 3)  # mine more than needed
        if args.llm_refine_topics:
            mined = refine_topics_with_llm(mined, args.auto_topics, args.language)
        else:
            mined = mined[:args.auto_topics]
        topics += mined

    # Normalize / dedup topics
    cleaned, seen = [], set()
    for t in topics:
        t = re.sub(r"\s+", " ", t).strip(" \t-—–")
        if len(t) < 6: continue
        key = t.lower()
        if key in seen: continue
        seen.add(key); cleaned.append(t)

    if not cleaned:
        raise SystemExit("No topics to process. Provide --topic/--topics-file or enable --auto-topics with sources.")

    # Cap workload
    if len(cleaned) > args.max_topics:
        print(f"ℹ️  {len(cleaned)} topics queued; limiting to first {args.max_topics}.")
        cleaned = cleaned[:args.max_topics]

    # 5) Generate per topic
    for i, topic in enumerate(cleaned, start=1):
        print(f"\n=== [{i}/{len(cleaned)}] {topic} ===")
        if registry.is_done(topic, args.language):
            print("↷ Skipping (already covered recently).")
            continue

        try:
            artifacts = generate_topic_artifacts(topic, args.language, kb)
        except Exception as e:
            print(f"⚠️  Generation failed for '{topic}': {e}")
            continue

        imgs_meta = []
        if PEXELS_API_KEY or UNSPLASH_ACCESS_KEY:
            selected = choose_images(topic, count=args.image_count)
            imgs_meta = download_images(selected, out_base / make_topic_id(topic) / "images")
        else:
            print("ℹ️  No image keys set; skipping images.")

        out_dir, topic_id = write_topic_bundle(out_base, topic, args.language, artifacts, imgs_meta)
        print(f"✓ Wrote bundle: {out_dir}")

        # robust registry updates (prevents KeyError: 'tally')
        try:
            registry.record(topic, topic_id, args.language)
            registry.save()
        except Exception as e:
            print(f"⚠️  Registry save failed (continuing): {e}")

    print("\n✅ Done.")
    print(f"Registry: {args.registry_path}")
    print(f"Index:    {args.index_out}")

if __name__ == "__main__":
    main()