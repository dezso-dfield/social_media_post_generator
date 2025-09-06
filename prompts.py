# src/llm/prompts.py

from typing import Dict

# ================================
# Global, local-LLM friendly system prompt
# ================================
BASE_SYSTEM = """You are a senior EU/Hungary-focused tech analyst and content strategist.
Follow these rules strictly:
1) Use only the facts provided in the input. If a detail is unknown, say so briefly.
2) Keep tone pragmatic and non-hyped. Prefer numbers, dates, standards, concrete vendors.
3) Default audience: Hungarian/EU decision-makers and practitioners. Briefly localize implications (EU AI Act, MiCA, GDPR, CVEs, local vendors).
4) Avoid vague claims and clichés. Prefer short paragraphs, scannable structure, and plain language.
5) If asked to include sources, cite them inline as [link] or list them at the bottom as “Sources”.
6) Never fabricate URLs, numbers, or quotations. No hallucinations.
7) If the user language is Hungarian (hu), write Hungarian that reads naturally for local professionals.
"""

# ================================
# Common platform guidance
# ================================
COMMON_STYLE = """Content quality guardrails:
- Be precise with dates and figures.
- Explain “why it matters” for EU/HU (AI Act, MiCA, cybersecurity norms).
- Offer actionable next steps for SMEs and enterprises.
- Prefer active voice and short sentences.
- Don’t oversell; be credible and specific.
"""

EU_LENS_HINT = """EU/Hungary lens tips:
- AI: reference AI Act obligations (risk, transparency, human oversight) when relevant.
- Blockchain: reference MiCA scope and disclosures, custody, stablecoin rules.
- Cybersecurity: reference CVEs, mitigations, SOC automation, NIS2 where useful.
"""

# ================================
# Topic guidance (used as “angle”)
# ================================
TOPIC_GUIDANCE: Dict[str, str] = {
    "ai": "Focus on AI Act readiness, model evaluations/assurance, on-device trends, enterprise rollout patterns.",
    "blockchain": "Focus on MiCA, tokenization and stablecoin rails, custody/treasury operations, enterprise integrations.",
    "cybersecurity": "Focus on incident learnings, CVEs, zero trust, SOC automation, practical mitigations."
}

# ================================
# Blog Post Template (Markdown)
# ================================
BLOG_POST_TEMPLATE = """{common_style}
{eu_lens}

Write a {words}-word **Markdown** news analysis blog post in {language}.

Inputs
- Title: {title}
- Source: {source}
- URL: {url}
- Summary: {summary}
- Angle (topic-specific): {angle}
- Audience: {audience}
- Brand voice: {brand_voice}

Structure (use these exact level-2 headings):
## TL;DR
- 2–3 bullet points, factual, no hype.

## What happened
- Key facts with dates and a one-sentence plain-English summary.

## Why it matters (EU/Hungary lens)
- Concrete implications for EU/HU orgs; mention AI Act / MiCA / NIS2 only if relevant.

## Context & numbers
- Benchmarks, adoption signals, vendor/model names, market size. If unknown, say “Data not available.”

## Risks & compliance notes
- Practical risks, obligations, mitigations. Keep it concise and useful.

## Actionable next steps
- 3–6 bullets: separate advice for SMEs and for enterprises.

Output rules
- Style: pragmatic, concise paragraphs (2–4 sentences each).
- Use inline links like [source] if you must reference the given URL.
- End with a short “Further reading” list if helpful (existing links only; no fabrications).
- At the very end, add an “Image queries:” line with 3–5 comma-separated search queries relevant to the story and Hungarian/EU context.

Begin now.
"""

# ================================
# X / Twitter Thread Template
# ================================
# Best practices:
# - 5–10 posts total; each <= 280 chars (aim 220–260).
# - Hook in post #1, CTA in last post.
# - Minimal emojis, 0–2 per post. 1–3 focused hashtags total (thread end).
# - Break up walls of text; avoid links until last/CTA unless crucial.
X_THREAD_TEMPLATE = """{common_style}
{eu_lens}

Write an X/Twitter thread in {language} with {post_count} posts about the topic.

Inputs
- Title: {title}
- Source: {source}
- URL: {url}
- Summary: {summary}
- Angle: {angle}
- Audience: {audience}
- Brand voice: {brand_voice}
- Desired CTA: {cta}

Thread rules
- Post 1: compelling hook in plain language (no clickbait). ≤ 280 chars.
- Posts 2–{last_main}: key insights (facts, numbers, EU/HU implications).
- One post must summarize “What it means for EU/HU” in 1–2 sentences.
- Final post: clear CTA (e.g., “Read full analysis”, “Subscribe”, or “Get the checklist”). Put the link only here if needed: {cta_link}
- Keep each post short (≤ 280 chars). 0–2 emojis max per post.
- Use at most 3 focused hashtags total; put them ONLY in the last post.

Output format EXACTLY:
1) <post text>
2) <post text>
...
{post_count}) <post text>
"""

# ================================
# LinkedIn Post Template
# ================================
# Best practices:
# - 900–1,400 characters total.
# - Strong scannability: short paragraphs, line breaks, 1–2 tasteful emojis max.
# - Add value first; CTA near the end; 2–4 relevant hashtags at the bottom.
LINKEDIN_POST_TEMPLATE = """{common_style}
{eu_lens}

Write a LinkedIn post in {language} (900–1400 characters) about the topic.

Inputs
- Title: {title}
- Source: {source}
- URL: {url}
- Summary: {summary}
- Angle: {angle}
- Audience: {audience}
- Brand voice: {brand_voice}
- Desired CTA: {cta}
- Optional mention(s): {mentions}   # e.g. @company or @author

Structure (don’t include headings, just line breaks):
- Hook (1–2 lines) with a clear value promise; no bait.
- 2–4 short paragraphs with the key facts and “why it matters” for EU/HU.
- 3–4 bullet points with practical next steps (mix SMEs/enterprises).
- CTA line with link: {cta_link}  (only once).
- Hashtags (2–4 focused): #AI #Blockchain #Cybersecurity (adjust to topic)

Rules
- Avoid jargon; make it readable on mobile.
- 0–2 emojis total (optional). Place near hook or CTA only.
- Do not exceed 1,400 characters.

Output: the post text only (ready to paste).
"""

# ================================
# Newsletter Template
# ================================
# Best practices:
# - Clear subject lines; useful preheader; modular body with links.
# - Scannable sections; one primary CTA; optional secondary link list.
NEWSLETTER_TEMPLATE = """{common_style}
{eu_lens}

Create a newsletter issue in {language} about the topic.

Inputs
- Title: {title}
- Source: {source}
- URL: {url}
- Summary: {summary}
- Angle: {angle}
- Audience: {audience}
- Brand voice: {brand_voice}
- Primary CTA: {cta}  ({cta_link})

Output format EXACTLY:

Subject line options (pick 3):
- [Option 1]
- [Option 2]
- [Option 3]

Preheader (≤ 110 chars):
- <one sentence>

Hero block:
- Headline (≤ 80 chars)
- 2–3 sentence dek explaining the value (EU/HU lens)
- Primary CTA: {cta} → {cta_link}

Main story:
- What happened (facts, dates)
- Why it matters (EU/HU implications; AI Act/MiCA/NIS2 only if relevant)
- 3–5 actionable takeaways

Quick hits (3 items):
- <item: one-sentence news + why it matters + source link if provided>
- <item>
- <item>

Further reading (existing links only, if any):
- <link text> – <URL>

Footer note (1–2 lines):
- Compliance or risk reminder relevant to the story (no legal advice).

End of output.
"""

# ================================
# Minimal renderer helper
# ================================
def render_prompt(kind: str, **kwargs) -> str:
    """
    kind: 'blog' | 'x_thread' | 'linkedin' | 'newsletter'
    kwargs expected across templates:
      language, title, source, url, summary, angle, audience, brand_voice,
      words (blog), post_count (x_thread), cta, cta_link, mentions
    """
    base = {
        "common_style": COMMON_STYLE,
        "eu_lens": EU_LENS_HINT,
        "language": kwargs.get("language", "en"),
        "title": kwargs.get("title", ""),
        "source": kwargs.get("source", ""),
        "url": kwargs.get("url", ""),
        "summary": kwargs.get("summary", ""),
        "angle": kwargs.get("angle", ""),
        "audience": kwargs.get("audience", "Hungarian/EU professionals"),
        "brand_voice": kwargs.get("brand_voice", "credible, pragmatic, numbers-first"),
        "cta": kwargs.get("cta", "Read the full analysis"),
        "cta_link": kwargs.get("cta_link", kwargs.get("url", "")),
        "mentions": kwargs.get("mentions", ""),
        "words": kwargs.get("words", 1100),
        "post_count": kwargs.get("post_count", 7),
    }

    if kind == "blog":
        return BLOG_POST_TEMPLATE.format(**base)
    if kind == "x_thread":
        # convenience var for template
        base["last_main"] = max(2, int(base["post_count"]) - 1)
        return X_THREAD_TEMPLATE.format(**base)
    if kind == "linkedin":
        return LINKEDIN_POST_TEMPLATE.format(**base)
    if kind == "newsletter":
        return NEWSLETTER_TEMPLATE.format(**base)
    raise ValueError(f"Unknown prompt kind: {kind}")