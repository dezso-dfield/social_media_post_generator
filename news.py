#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py — Local (Ollama) multichannel content generator with free web search + site scraping.
Now includes a persistent registry to skip already-covered articles and tally topics.

Inputs you can mix & match:
  A) --url URL ...                # scrape one or more URLs directly
  B) --sites sites.yaml           # crawl site start pages, pick article links, then scrape
  C) --q "query" --use-free-search   # DuckDuckGo Lite + GDELT to find links, then scrape
  D) --input-json article.json    # one prepared article dict: {title, link, source, summary, topic}

Outputs per article (in output/<post_id>/):
  - post.md              (Markdown + YAML front matter + hero image)
  - twitter_thread.txt   (validated ≤280 chars per post)
  - linkedin.txt         (validated ≤1400 chars total)
  - newsletter.md
  - manifest.json
  - images/img_*.jpg     (Pexels/Unsplash; optional if keys present)

Plus a persistent registry (default: output/registry.json) to avoid duplicates and track topic totals.

Environment: configure in .env (OLLAMA_BASE, OLLAMA_MODEL, OLLAMA_TEMPERATURE, PEXELS_API_KEY, UNSPLASH_ACCESS_KEY)
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
import sys
import time
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse, urlencode

import feedparser
import requests
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ----- load .env -----
load_dotenv()  # reads .env from project root

# ======================
# Environment
# ======================
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.4"))
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

UA_HDRS = {"User-Agent": "Mozilla/5.0 (compatible; DFieldBot/1.0)"}

# -----------------------
# Utilities
# -----------------------

def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)

def make_post_id(title: str, url: str) -> str:
    s = slugify(title)[:60]
    date = dt.datetime.now(timezone.utc).strftime("%Y%m%d")
    h = hashlib.sha1((title + "|" + (url or "")).encode("utf-8")).hexdigest()[:8]
    return f"{date}-{s}-{h}"

def to_abs(base: str, href: str) -> str:
    try:
        return urljoin(base, href)
    except Exception:
        return href

# -----------------------
# Prompts (platform-optimized)
# -----------------------

# -----------------------
# Prompts (platform-optimized)
# -----------------------

BASE_SYSTEM = """You are a senior technical analyst and instructor.

Follow strictly:
1) Use only the provided input. If something is unknown, write: "Not in provided sources."
2) Be precise and pragmatic. Prefer standards, dates, commands, and vendor-neutral guidance.
3) Audience: experienced practitioners and engineering leaders. Assume basic literacy; avoid hand-holding.
4) Short sentences. No filler. No hype. No clichés.
5) Bulleted lists must use "- " bullets. Use numbered lists only when the order truly matters.
6) Prefer concrete code, configs, commands, and diagnostics over generic advice.
7) Never fabricate URLs, numbers, or quotes.
"""

COMMON_STYLE = """Style guardrails:

- Lead with the core idea in plain language, then deepen with specifics.
- Use "- " bullets for tight points (one idea each).
- Include copy-pasteable snippets (CLI, config, code) where useful.
- Call out failure modes and verification steps.
- End with brief "Next steps" for individual contributors and for teams.
- Do not write blog tropes like "In this article".
"""

# Keep the variable to avoid NameError where it’s interpolated; make it empty.
EU_LENS_HINT = ""

TOPIC_GUIDANCE: Dict[str, str] = {
    "ai": "Focus on real-world deployments, evaluation, observability, and cost/perf tradeoffs.",
    "blockchain": "Focus on custody, tokenization, stable settlement rails, and enterprise integration patterns.",
    "cybersecurity": "Focus on incidents, CVEs, zero trust, SOC automation, and practical mitigations."
}

def topic_angle(topic: str) -> str:
    return TOPIC_GUIDANCE.get(topic, "")

BLOG_POST_TEMPLATE = f"""{COMMON_STYLE}
{EU_LENS_HINT}

Write a {{words}}-word **Markdown** news analysis post in {{language}}.

Inputs
- Title: {{title}}
- Source: {{source}}
- URL: {{url}}
- Summary: {{summary}}
- Angle: {{angle}}
- Audience: {{audience}}
- Brand voice: {{brand_voice}}

Structure (use these exact level-2 headings):
## TL;DR
- 2–3 bullets. Facts only.

## What happened
- Key facts with dates and a one-sentence plain-language summary.

## Why it matters
- Concrete implications for builders and decision-makers.

## Context & numbers
- Benchmarks, adoption signals, vendor/model names, market size. If unknown: "Not in provided sources."

## Risks & tradeoffs
- Practical risks, obligations, mitigations. Keep it useful.

## Actionable next steps
- 3–6 bullets. Separate advice for ICs and for teams.

Output rules
- Style: concise paragraphs (2–4 sentences each). No fluff.
- Use inline links like [source] if referencing the given URL.
- End with a short "Further reading" list if helpful (existing links only; no fabrications).
- At the very end, add an "Image queries:" line with 3–5 comma-separated search queries relevant to the story.

Begin now.
"""

X_THREAD_TEMPLATE = f"""{COMMON_STYLE}
{EU_LENS_HINT}

Write an X/Twitter thread in {{language}} with {{post_count}} posts.

Inputs
- Title: {{title}}
- Source: {{source}}
- URL: {{url}}
- Summary: {{summary}}
- Angle: {{angle}}
- Audience: {{audience}}
- Brand voice: {{brand_voice}}
- Desired CTA: {{cta}}

Thread rules
- Post 1: compelling hook in plain language (no clickbait). ≤ 280 chars.
- Posts 2–{{last_main}}: key insights (facts, numbers, implications).
- One post should summarize "What it means for teams" in 1–2 sentences.
- Final post: clear CTA. Put the link only here: {{cta_link}}
- Keep each post ≤ 280 chars. Use at most 3 focused hashtags total, ONLY in the last post.
- 0–2 emojis max per post (optional).

Output format EXACTLY:
1) <post text>
2) <post text>
...
{{post_count}}) <post text>
"""

LINKEDIN_POST_TEMPLATE = f"""{COMMON_STYLE}
{EU_LENS_HINT}

Write a LinkedIn post in {{language}} (900–1400 characters).

Inputs
- Title: {{title}}
- Source: {{source}}
- URL: {{url}}
- Summary: {{summary}}
- Angle: {{angle}}
- Audience: {{audience}}
- Brand voice: {{brand_voice}}
- Desired CTA: {{cta}}
- Optional mention(s): {{mentions}}

Structure (use line breaks, no headings):
- Hook (1–2 lines) with a clear value promise; no bait.
- 2–4 short paragraphs with key facts and why it matters.
- 3–4 "- " bullets with practical next steps (mix ICs/teams).
- CTA line with link (once): {{cta_link}}
- 2–4 focused hashtags at the end.

Rules
- Readable on mobile. Minimal jargon.
- 0–2 emojis total (optional, near hook or CTA).
- Hard cap 1,400 characters.

Output: the post text only.
"""

NEWSLETTER_TEMPLATE = f"""{COMMON_STYLE}
{EU_LENS_HINT}

Create a newsletter issue in {{language}} about the topic.

Inputs
- Title: {{title}}
- Source: {{source}}
- URL: {{url}}
- Summary: {{summary}}
- Angle: {{angle}}
- Audience: {{audience}}
- Brand voice: {{brand_voice}}
- Primary CTA: {{cta}}  ({{cta_link}})

Output format EXACTLY:

Subject line options (pick 3):
- [Option 1]
- [Option 2]
- [Option 3]

Preheader (≤ 110 chars):
- <one sentence>

Hero block:
- Headline (≤ 80 chars)
- 2–3 sentence dek explaining the value
- Primary CTA: {{cta}} → {{cta_link}}

Main story:
- What happened (facts, dates)
- Why it matters (implications)
- 3–5 actionable takeaways

Quick hits (3 items):
- <item: one-sentence news + why it matters + source link if provided>
- <item>
- <item>

Further reading (existing links only, if any):
- <link text> – <URL>

Footer note (1–2 lines):
- A concise risk or operations reminder (no legal advice).

End of output.
"""

def render_prompt(kind: str, **kwargs) -> str:
    base = dict(
        language=kwargs.get("language", "en"),
        title=kwargs.get("title", ""),
        source=kwargs.get("source", ""),
        url=kwargs.get("url", ""),
        summary=kwargs.get("summary", ""),
        angle=kwargs.get("angle", ""),
        audience=kwargs.get("audience", "practitioners and engineering leaders"),
        brand_voice=kwargs.get("brand_voice", "credible, pragmatic, numbers-first"),
        cta=kwargs.get("cta", "Read the full analysis"),
        cta_link=kwargs.get("cta_link", kwargs.get("url", "")),
        mentions=kwargs.get("mentions", ""),
        words=kwargs.get("words", 1100),
        post_count=kwargs.get("post_count", 7),
    )
    if kind == "blog":
        return BLOG_POST_TEMPLATE.format(**base)
    if kind == "x_thread":
        base["last_main"] = max(2, int(base["post_count"]) - 1)
        return X_THREAD_TEMPLATE.format(**base)
    if kind == "linkedin":
        return LINKEDIN_POST_TEMPLATE.format(**base)
    if kind == "newsletter":
        return NEWSLETTER_TEMPLATE.format(**base)
    raise ValueError(f"Unknown prompt kind: {kind}")

# -----------------------
# Ollama (robust: HTTP shim → native chat → generate → CLI)
# -----------------------

def _http_chat_completions(url: str, headers: Dict[str, str], model: str, system: str, user_prompt: str,
                           temperature: float, timeout: int = 120) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise requests.HTTPError(f"{resp.status_code} {resp.reason}", response=resp)
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected LLM response (OpenAI-compatible): {data}")

def _ollama_chat_native(base: str, model: str, system: str, user_prompt: str,
                        temperature: float, timeout: int = 120) -> str:
    url = f"{base.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": float(temperature)},
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected LLM response (Ollama /api/chat): {data}")

def _ollama_generate_native(base: str, model: str, prompt: str,
                            temperature: float, timeout: int = 120) -> str:
    url = f"{base.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": float(temperature)},
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["response"]
    except Exception:
        raise RuntimeError(f"Unexpected LLM response (Ollama /api/generate): {data}")

def _ollama_cli_run(model: str, prompt: str, timeout: int = 600) -> str:
    if not shutil.which("ollama"):
        raise RuntimeError("`ollama` CLI not found. Install Ollama or add to PATH.")
    combined = f"System:\n{BASE_SYSTEM}\n\nUser:\n{prompt}"
    proc = subprocess.run(
        ["ollama", "run", model],
        input=combined.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Ollama CLI error: {proc.stderr.decode('utf-8', 'ignore')}")
    return proc.stdout.decode("utf-8", "ignore").strip()

def call_ollama(prompt: str) -> str:
    base = os.environ.get("OLLAMA_BASE", OLLAMA_BASE)
    model = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)
    temp = float(os.environ.get("OLLAMA_TEMPERATURE", str(OLLAMA_TEMPERATURE)))

    # 1) OpenAI-compatible
    try:
        url = f"{base.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        return _http_chat_completions(url, headers, model, BASE_SYSTEM, prompt, temp)
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status not in (404, 405, 501):
            body = e.response.text if getattr(e, "response", None) else ""
            raise RuntimeError(f"Ollama OpenAI-compatible call failed ({status}): {body}") from e
    except Exception:
        pass

    # 2) Native /api/chat
    try:
        return _ollama_chat_native(base, model, BASE_SYSTEM, prompt, temp)
    except Exception:
        pass

    # 3) Native /api/generate
    try:
        combined = f"System:\n{BASE_SYSTEM}\n\nUser:\n{prompt}"
        return _ollama_generate_native(base, model, combined, temp)
    except Exception:
        pass

    # 4) CLI fallback
    return _ollama_cli_run(model, prompt)

# -----------------------
# Free Search (DuckDuckGo Lite + GDELT)
# -----------------------

def search_duckduckgo(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    url = "https://duckduckgo.com/html/?" + urlencode({"q": query})
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    for a in soup.select("a.result__a")[:limit]:
        link = a.get("href")
        title = a.get_text(" ", strip=True)
        out.append({"title": title, "link": link, "summary": "", "source": "DuckDuckGo", "published": ""})
    return out

def search_gdelt(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    params = {"query": query, "maxrecords": limit, "format": "json", "sort": "DateDesc"}
    r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    out = []
    for item in data.get("articles", [])[:limit]:
        out.append({
            "title": item.get("title") or "",
            "link": item.get("url") or "",
            "summary": item.get("seendate") or "",
            "source": item.get("sourceCommonName") or "GDELT",
            "published": item.get("seendate") or "",
        })
    return out

def free_news_for_topic(topic: str, limit: int = 6) -> List[Dict[str, Any]]:
    q = {
        "ai": "artificial intelligence EU AI Act",
        "blockchain": "blockchain MiCA EU stablecoin",
        "cybersecurity": "cybersecurity NIS2 CVE EU"
    }.get(topic, topic)
    ddg = search_duckduckgo(q, limit=limit)
    gdelt = search_gdelt(q, limit=limit)
    seen, merged = set(), []
    for it in ddg + gdelt:
        u = it.get("link")
        if not u or u in seen:
            continue
        seen.add(u)
        merged.append(it)
    return merged[:limit]

# -----------------------
# Scraping (generic)
# -----------------------

def fetch_html(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=UA_HDRS, timeout=30)
        if r.status_code >= 400:
            return None
        return r.text
    except Exception:
        return None

def extract_title(soup: BeautifulSoup) -> str:
    for sel in [
        'meta[property="og:title"]', 'meta[name="twitter:title"]',
    ]:
        tag = soup.select_one(sel)
        if tag and tag.get("content"):
            return tag["content"].strip()
    if soup.title and soup.title.text:
        return soup.title.text.strip()
    h1 = soup.find("h1")
    return h1.get_text(" ", strip=True) if h1 else ""

def extract_published(soup: BeautifulSoup) -> str:
    for sel in [
        'meta[property="article:published_time"]',
        'meta[name="pubdate"]',
        'meta[name="date"]',
        'time[datetime]',
    ]:
        tag = soup.select_one(sel)
        if tag:
            if tag.has_attr("content"):
                return tag["content"].strip()
            if tag.has_attr("datetime"):
                return tag["datetime"].strip()
            return tag.get_text(" ", strip=True)
    return ""

def extract_summary(soup: BeautifulSoup) -> str:
    for sel in [
        'meta[name="description"]', 'meta[property="og:description"]', 'meta[name="twitter:description"]'
    ]:
        tag = soup.select_one(sel)
        if tag and tag.get("content"):
            return tag["content"].strip()
    # fall back: first long paragraph
    for p in soup.find_all("p"):
        txt = p.get_text(" ", strip=True)
        if len(txt) > 140:
            return txt[:400]
    return ""

def extract_main_text(soup: BeautifulSoup) -> str:
    # Prefer <article>, then obvious main containers
    candidates = []
    for sel in ["article", "main", "#content", ".content", ".post", ".article__body", ".entry-content"]:
        for node in soup.select(sel):
            candidates.append(node)
    if not candidates:
        candidates = [soup.body or soup]

    # score nodes by total text in <p> children
    best_text = ""
    best_len = 0
    for node in candidates:
        parts = []
        for p in node.find_all("p"):
            t = p.get_text(" ", strip=True)
            if len(t) >= 40:
                parts.append(t)
        text = "\n\n".join(parts)
        if len(text) > best_len:
            best_text, best_len = text, len(text)

    if best_len < 200:
        parts = []
        for p in soup.find_all("p"):
            t = p.get_text(" ", strip=True)
            if len(t) >= 40:
                parts.append(t)
        best_text = "\n\n".join(parts)
    return best_text.strip()

def scrape_url(url: str) -> Optional[Dict[str, Any]]:
    html_doc = fetch_html(url)
    if not html_doc:
        return None
    soup = BeautifulSoup(html_doc, "html.parser")
    return {
        "title": extract_title(soup) or url,
        "link": url,
        "source": urlparse(url).netloc,
        "summary": extract_summary(soup),
        "published": extract_published(soup),
        "content": extract_main_text(soup),
    }

def crawl_site(start_url: str, link_selector: Optional[str], limit: int = 5) -> List[str]:
    html_doc = fetch_html(start_url)
    if not html_doc:
        return []
    soup = BeautifulSoup(html_doc, "html.parser")
    links = set()
    if link_selector:
        for a in soup.select(link_selector)[:limit * 4]:
            href = a.get("href")
            if not href:
                continue
            links.add(to_abs(start_url, href))
    else:
        # heuristic: article-like links
        for a in soup.find_all("a", href=True)[:limit * 10]:
            href = a["href"]
            abs_url = to_abs(start_url, href)
            if re.search(r"/20\d{2}/|/news/|/story/|/article/", abs_url) or any(k in abs_url.lower() for k in ["ai", "block", "cyber", "security"]):
                links.add(abs_url)
    # keep within domain
    base_host = urlparse(start_url).netloc
    links = [u for u in links if urlparse(u).netloc.endswith(base_host.split(":")[0])]
    return links[:limit]

# -----------------------
# Images
# -----------------------

def search_pexels(query: str, api_key: str, per_page: int = 3) -> List[Dict[str, Any]]:
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": per_page}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    out = []
    for p in data.get("photos", []):
        out.append({"url": p["src"]["large"], "photographer": p.get("photographer"), "credit_url": p.get("url"), "provider": "Pexels"})
    return out

def search_unsplash(query: str, access_key: str, per_page: int = 3) -> List[Dict[str, Any]]:
    url = "https://api.unsplash.com/search/photos"
    params = {"query": query, "per_page": per_page, "client_id": access_key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    out = []
    for p in data.get("results", []):
        link = p.get("urls", {}).get("regular") or p.get("urls", {}).get("full")
        credit = p.get("user", {}).get("links", {}).get("html")
        out.append({"url": link, "photographer": p.get("user", {}).get("name"), "credit_url": credit, "provider": "Unsplash"})
    return out

def download_images(imgs: List[Dict[str, Any]], dest_dir: Path) -> List[Dict[str, Any]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, im in enumerate(imgs, start=1):
        path = dest_dir / f"img_{i}.jpg"
        try:
            r = requests.get(im["url"], timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            path.write_bytes(r.content)
            meta = dict(im)
            meta["local_path"] = str(Path("images") / path.name)
            out.append(meta)
            print(f"🖼️  Saved image: {path.name}  ({im.get('provider','?')}, {im.get('photographer','Unknown')})")
        except Exception as e:
            print(f"⚠️  Image download failed for {im.get('url')}: {e}")
            continue
    return out

def choose_images(queries: List[str], count: int = 3) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pexels_key = PEXELS_API_KEY or os.getenv("PEXELS_API_KEY")
    unsplash_key = UNSPLASH_ACCESS_KEY or os.getenv("UNSPLASH_ACCESS_KEY")
    if not pexels_key and not unsplash_key:
        print("⚠️  No PEXELS_API_KEY or UNSPLASH_ACCESS_KEY set; skipping image search.")
        return out
    for q in queries or []:
        if len(out) >= count:
            break
        results: List[Dict[str, Any]] = []
        try:
            if pexels_key:
                results.extend(search_pexels(q, pexels_key, per_page=2))
            if unsplash_key and len(results) < 1:
                results.extend(search_unsplash(q, unsplash_key, per_page=2))
        except Exception as e:
            print(f"⚠️  Image search error for '{q}': {e}")
        if results:
            out.append(results[0])
        else:
            print(f"ℹ️  No image found for query: {q}")
    if not out:
        print("ℹ️  No images selected after all queries.")
    return out[:count]

# -----------------------
# Rendering & validation
# -----------------------

def extract_image_queries_from_blog(blog_md: str, meta: Dict[str, Any]) -> List[str]:
    # 1) Try to read the explicit block the prompt asks for
    lines = [l.strip() for l in blog_md.splitlines() if l.strip()]
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].lower().startswith("image queries:"):
            q = lines[i].split(":", 1)[1].strip()
            qs = [x.strip() for x in q.split(",") if x.strip()]
            if qs:
                return qs[:5]

    # 2) Fallback: build queries from meta/title/topic
    title = meta.get("title", "")
    topic = (meta.get("topic") or "").lower()
    base = [title, f"{title} EU", f"{title} Hungary"]
    if topic == "ai":
        base += ["AI Act compliance", "AI model demo", "enterprise AI meeting"]
    elif topic == "blockchain":
        base += ["MiCA crypto policy", "stablecoin EU", "tokenization finance EU"]
    elif topic == "cybersecurity":
        base += ["SOC operations", "NIS2 compliance", "security operations center"]
    # de-dup and trim
    seen, out = set(), []
    for q in base:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
        if len(out) >= 5:
            break
    return out

def render_markdown(meta: Dict[str, Any], body_md: str, images: List[Dict[str, Any]]) -> str:
    hero = images[0] if images else None
    fm_obj = {
        "title": meta["title"],
        "slug": slugify(meta["title"]),
        "date": meta.get("date") or dt.datetime.now(timezone.utc).isoformat(),
        "tags": meta.get("tags", []),
        "topic": meta.get("topic"),
        "language": meta.get("language", "en"),
        "source_url": meta.get("source_url"),
        "source_name": meta.get("source_name"),
        "hero_image": hero.get("local_path") if hero else None,
        "images": images,
        "summary": meta.get("summary", "")
    }
    fm = "---\n" + yaml.safe_dump(fm_obj, sort_keys=False, allow_unicode=True) + "---\n\n"
    hero_block = ""
    if hero:
        cap = f'![{html.escape(fm_obj["title"])}]({hero["local_path"]})\n*Photo: {hero.get("photographer","Unknown")} via {hero.get("provider")}*\n\n'
        hero_block = cap
    return fm + hero_block + body_md.strip() + "\n"

def validate_x_thread(text: str, post_count: int) -> str:
    posts = []
    for m in re.finditer(r"^\s*(\d+)\)\s*(.+)$", text, flags=re.M):
        posts.append((int(m.group(1)), m.group(2)))
    if not posts:
        raw = [l.strip() for l in text.strip().splitlines() if l.strip()]
        posts = list(enumerate(raw[:post_count], start=1))
    posts = posts[:post_count]
    if len(posts) < post_count:
        for n in range(len(posts) + 1, post_count + 1):
            posts.append((n, "…"))
    def clip(s: str, limit: int = 280) -> str:
        return s if len(s) <= limit else (s[:277] + "…")
    lines = [f"{idx}) {clip(txt)}" for idx, txt in posts]
    return "\n".join(lines)

def validate_linkedin(text: str, max_chars: int = 1400) -> str:
    s = re.sub(r"\s+\n", "\n", text.strip())
    s = re.sub(r"\n{3,}", "\n\n", s)
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    last_dot = cut.rfind(".")
    if last_dot > 600:
        return cut[:last_dot + 1] + " …"
    return cut.rstrip() + " …"

# -----------------------
# Generation orchestration (Ollama)
# -----------------------

def generate_all_channels(article: Dict[str, Any], language: str, words: int) -> Dict[str, str]:
    angle = topic_angle(article.get("topic", ""))
    base_kwargs = dict(
        language=language,
        title=article.get("title", ""),
        source=article.get("source", ""),
        url=article.get("link", ""),
        summary=article.get("summary", ""),
        angle=angle,
        audience="Hungarian/EU decision-makers",
        brand_voice="credible, pragmatic, numbers-first",
    )
    # Blog
    blog_prompt = render_prompt("blog", **base_kwargs, words=words)
    blog_md = call_ollama(blog_prompt)

    # X thread
    x_prompt = render_prompt("x_thread", **base_kwargs, cta="Read the full analysis",
                             cta_link=article.get("link", ""), post_count=7)
    x_thread_raw = call_ollama(x_prompt)
    x_thread = validate_x_thread(x_thread_raw, post_count=7)

    # LinkedIn
    li_prompt = render_prompt("linkedin", **base_kwargs, cta="Full write-up",
                              cta_link=article.get("link", ""), mentions="")
    linkedin_raw = call_ollama(li_prompt)
    linkedin = validate_linkedin(linkedin_raw)

    # Newsletter
    nl_prompt = render_prompt("newsletter", **base_kwargs, cta="Read the full analysis",
                              cta_link=article.get("link", ""))
    newsletter_md = call_ollama(nl_prompt)

    return {
        "blog_md": blog_md,
        "x_thread": x_thread,
        "linkedin": linkedin,
        "newsletter_md": newsletter_md
    }

# -----------------------
# Feed mode (RSS / free search)
# -----------------------

def _within_days(published_parsed, days_back: Optional[int]) -> bool:
    if not published_parsed or days_back is None:
        return True
    pub = dt.datetime(*published_parsed[:6], tzinfo=timezone.utc)
    return pub >= (dt.datetime.now(timezone.utc) - dt.timedelta(days=days_back))

def fetch_from_rss(rss_urls: List[str], days_back: Optional[int] = None, limit: int = 10) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for url in rss_urls:
        feed = feedparser.parse(url)
        for e in feed.entries[:limit]:
            if not _within_days(getattr(e, "published_parsed", None), days_back):
                continue
            items.append({
                "title": getattr(e, "title", ""),
                "summary": getattr(e, "summary", ""),
                "link": getattr(e, "link", ""),
                "published": getattr(e, "published", ""),
                "source": feed.feed.get("title", url),
            })
    seen, dedup = set(), []
    for it in items:
        if it["title"] in seen:
            continue
        seen.add(it["title"])
        dedup.append(it)
    return dedup[:limit]

def free_news_for_topics(topics: Iterable[str], per_topic: int = 2) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []
    for t in topics:
        all_items.extend(free_news_for_topic(t, limit=per_topic))
    return all_items

# -----------------------
# Registry: avoid duplicates & track topics
# -----------------------

@dataclass
class RegistryItem:
    title: str
    link: str
    topic: str
    post_id: str
    created_at: str  # ISO8601 UTC

class TopicRegistry:
    def __init__(self, path: Path, dedupe_days: int = 60):
        self.path = path
        self.dedupe_days = dedupe_days
        self.data = {
            "topics_tally": {"ai": 0, "blockchain": 0, "cybersecurity": 0},
            "seen": {},       # fingerprint -> ISO date
            "items": []       # List[RegistryItem dict]
        }
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                # back up corrupted registry, start fresh
                try:
                    self.path.replace(self.path.with_suffix(".bak"))
                except Exception:
                    pass
                self.data = {"topics_tally": {"ai": 0, "blockchain": 0, "cybersecurity": 0}, "seen": {}, "items": []}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _norm_title(t: str) -> str:
        t = (t or "").strip().lower()
        t = re.sub(r"\s+", " ", t)
        return t[:200]

    @staticmethod
    def _norm_link(u: str) -> str:
        try:
            p = urlparse(u or "")
            return f"{p.netloc.lower()}{p.path}"
        except Exception:
            return (u or "").lower()

    def fingerprint(self, title: str, link: str) -> str:
        base = f"{self._norm_title(title)}|{self._norm_link(link)}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def _is_fresh(self, iso: str) -> bool:
        if self.dedupe_days <= 0:
            return True
        try:
            then = dt.datetime.fromisoformat(iso.replace("Z", "")).replace(tzinfo=timezone.utc)
            return then >= (dt.datetime.now(timezone.utc) - dt.timedelta(days=self.dedupe_days))
        except Exception:
            return True

    def is_done(self, title: str, link: str) -> bool:
        fp = self.fingerprint(title, link)
        iso = self.data["seen"].get(fp)
        return bool(iso and self._is_fresh(iso))

    def record(self, title: str, link: str, topic: str, post_id: str):
        now_iso = dt.datetime.now(timezone.utc).isoformat()
        fp = self.fingerprint(title, link)
        self.data["seen"][fp] = now_iso
        if topic not in self.data["topics_tally"]:
            self.data["topics_tally"][topic] = 0
        self.data["topics_tally"][topic] += 1
        self.data["items"].append(asdict(RegistryItem(
            title=title, link=link, topic=topic, post_id=post_id, created_at=now_iso
        )))

# -----------------------
# IO helpers
# -----------------------

def write_bundle(out_base: Path, post_id: str, files: Dict[str, str], manifest: Dict[str, Any]) -> Path:
    out_dir = out_base / post_id
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (out_dir / name).write_text(content, encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Multichannel news-to-content (Ollama + free search + scraper)")
    g_in = ap.add_argument_group("Inputs")
    g_in.add_argument("--input-json", help="One article JSON {title,link,source,summary,topic}")
    g_in.add_argument("--url", action="append", help="Scrape a specific URL (can repeat)")
    g_in.add_argument("--sites", help="sites.yaml with site start pages & optional link_selector")
    g_in.add_argument("--q", action="append", help="Free search query (DuckDuckGo + GDELT), can repeat")
    g_in.add_argument("--topics", nargs="*", default=[], help="Shortcut for topic queries: ai blockchain cybersecurity")
    g_in.add_argument("--feeds", help="feeds.yaml for RSS fallback")
    g_in.add_argument("--use-free-search", action="store_true", help="Use free search for topics/queries")

    g_out = ap.add_argument_group("Outputs")
    g_out.add_argument("--output-dir", default="output")
    g_out.add_argument("--language", default="en")
    g_out.add_argument("--min-words", type=int, default=900)
    g_out.add_argument("--max-words", type=int, default=1400)
    g_out.add_argument("--image-count", type=int, default=3)
    g_out.add_argument("--registry-path", default="output/registry.json",
                       help="Where to store processed items & topic tallies")
    g_out.add_argument("--dedupe-days", type=int, default=60,
                       help="Skip if an article (title+link) was already done within N days (0 = always skip if seen)")
    g_out.add_argument("--max-articles", type=int, default=10,
                       help="Process at most N articles (after de-dup)")

    args = ap.parse_args()
    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    words = max(args.min_words, min(args.max_words, args.min_words + 200))

    registry = TopicRegistry(Path(args.registry_path), dedupe_days=args.dedupe_days)

    # Collect candidate article dicts (title/link/source/summary/topic)
    candidates: List[Dict[str, Any]] = []

    # A) input-json
    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            a = json.load(f)
        for k in ["title", "link", "source", "summary", "topic"]:
            if k not in a:
                raise SystemExit(f"--input-json is missing: {k}")
        candidates.append(a)

    # B) explicit URLs
    if args.url:
        for u in args.url:
            art = scrape_url(u)
            if art:
                # guess topic by keywords
                t = "ai" if "ai" in u.lower() else "blockchain" if "block" in u.lower() else "cybersecurity" if "cyber" in u.lower() or "security" in u.lower() else "ai"
                art["topic"] = t
                candidates.append(art)

    # C) sites.yaml crawl
    if args.sites:
        sites_cfg = load_yaml(args.sites)
        for s in sites_cfg.get("sites", []):
            start = s.get("start_url")
            sel = s.get("link_selector")
            lim = int(s.get("limit", 5))
            if not start:
                continue
            links = crawl_site(start, sel, limit=lim)
            for u in links:
                art = scrape_url(u)
                if art and len((art.get("content") or "")) > 500:
                    art["topic"] = s.get("topic") or ("ai" if "ai" in u.lower() else "blockchain" if "block" in u.lower() else "cybersecurity" if "cyber" in u.lower() or "security" in u.lower() else "ai")
                    candidates.append(art)

    # D) free search
    if args.use_free_search and (args.q or args.topics):
        if args.q:
            for query in args.q:
                for item in search_duckduckgo(query, limit=5) + search_gdelt(query, limit=5):
                    art = scrape_url(item["link"])
                    if art and len((art.get("content") or "")) > 500:
                        art["topic"] = "ai"  # neutral default; adjust below
                        if any(k in art["link"].lower() for k in ["block", "defi", "stablecoin", "token"]):
                            art["topic"] = "blockchain"
                        elif any(k in art["link"].lower() for k in ["cyber", "cve", "ransom", "security"]):
                            art["topic"] = "cybersecurity"
                        candidates.append(art)
        if args.topics:
            topic_items = free_news_for_topics(args.topics, per_topic=2)
            for it in topic_items:
                art = scrape_url(it["link"])
                if art and len((art.get("content") or "")) > 500:
                    art["topic"] = "ai"
                    if "block" in art["link"].lower(): art["topic"] = "blockchain"
                    if "cyber" in art["link"].lower() or "security" in art["link"].lower(): art["topic"] = "cybersecurity"
                    candidates.append(art)

    # E) feeds.yaml (RSS fallback)
    if args.feeds:
        feeds_cfg = load_yaml(args.feeds)
        for topic, node in feeds_cfg.get("topics", {}).items():
            rss = node.get("rss", [])
            for it in fetch_from_rss(rss, days_back=5, limit=6):
                art = scrape_url(it["link"])
                if art and len((art.get("content") or "")) > 500:
                    art["topic"] = topic
                    art["title"] = art["title"] or it["title"]
                    art["summary"] = art["summary"] or it["summary"]
                    candidates.append(art)

    if not candidates:
        raise SystemExit("No articles found. Try --url, --sites, or --use-free-search with --q/--topics.")

    # Deduplicate by link
    seen = set()
    unique: List[Dict[str, Any]] = []
    for a in candidates:
        u = a.get("link")
        if not u or u in seen:
            continue
        seen.add(u)
        unique.append(a)

    total = len(unique)
    limit = max(1, int(args.max_articles))
    if total > limit:
        print(f"ℹ️  {total} articles queued; limiting to first {limit}.")
        unique = unique[:limit]

    # Generate content bundles
    for idx, art in enumerate(unique, start=1):
        title = art.get("title", "") or art.get("link", "")
        print(f"\n=== [{idx}/{len(unique)}] {title[:90]} ===")

        # Skip if already processed recently
        if registry.is_done(art["title"], art["link"]):
            print(f"↷ Skipping (already covered recently): {art['title'][:90]}")
            continue

        # If the scraper captured body text, append 1-2 line summary for the model
        if art.get("content") and not art.get("summary"):
            art["summary"] = art["content"][:400]

        # --- Call Ollama to generate all channels ---
        bundle = generate_all_channels(
            {"title": art["title"], "link": art["link"], "source": art["source"], "summary": art.get("summary", ""), "topic": art.get("topic", "ai")},
            language=args.language,
            words=words
        )

        # --- Images ---
        post_id = make_post_id(art["title"], art["link"])
        post_dir = Path(args.output_dir) / post_id
        img_dir = post_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "title": art["title"],
            "topic": art.get("topic", "ai"),
            "language": args.language,
            "source_url": art["link"],
            "source_name": art["source"],
            "summary": art.get("summary", ""),
            "tags": [art.get("topic", "news"), "news"]
        }

        queries = extract_image_queries_from_blog(bundle["blog_md"], meta)
        print(f"🔎 Image queries: {queries[:3]}{' …' if len(queries)>3 else ''}")

        selected = choose_images(queries, count=args.image_count)
        downloaded = download_images(selected, img_dir)
        blog_full_md = render_markdown(meta, bundle["blog_md"], downloaded)

        files = {
            "post.md": blog_full_md,
            "twitter_thread.txt": bundle["x_thread"],
            "linkedin.txt": bundle["linkedin"],
            "newsletter.md": bundle["newsletter_md"],
        }
        manifest = {
            "post_id": post_id,
            "meta": meta,
            "images": downloaded,
            "image_queries": queries,
            "source_article": {k: art.get(k) for k in ["title","link","source","summary","published","topic"]},
            "ollama_model": os.getenv("OLLAMA_MODEL", OLLAMA_MODEL),
        }
        out_dir = write_bundle(Path(args.output_dir), post_id, files, manifest)
        print(f"✓ Wrote bundle: {out_dir}")

        # Record in registry and save
        registry.record(art["title"], art["link"], art.get("topic","ai"), post_id)
        registry.save()

    tally = registry.data.get("topics_tally", {})
    print(f"\nTopic totals → AI: {tally.get('ai',0)} | Blockchain: {tally.get('blockchain',0)} | Cybersecurity: {tally.get('cybersecurity',0)}")
    print("Done.")

if __name__ == "__main__":
    main()