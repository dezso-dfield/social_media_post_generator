# Local AI Content Toolkit (Ollama) — README

Generate clean, practical educational and news content from **your own KB + scraped sources**, all on your machine using **Ollama-compatible** models. Two tools are included:

- `education.py` — builds **educational artifacts** per topic (tutorial, methodology, checklist, slides outline, LinkedIn post) from a local KB plus optional URLs/RSS/sites.
- `main.py` — turns **news articles** (scraped or found via free search/RSS) into a multi-channel bundle (blog post, X/Twitter thread, LinkedIn post, newsletter) with images and a persistent registry.

Both scripts are deterministic, file-system first, and avoid hypey AI tone by design.

---

## 1) Installation

# Python 3.10+ recommended
python3 -m venv .venv
source .venv/bin/activate

pip install -U \
  beautifulsoup4 \
  python-dotenv \
  pyyaml \
  requests \
  pdfminer.six \
  feedparser

Optional: pdfminer.six is only needed if you want to parse PDFs; it’s safe to omit.

⸻

2) Configure .env

Create a .env next to your scripts:

# Ollama (or any OpenAI-compatible shim)
OLLAMA_BASE=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
OLLAMA_TEMPERATURE=0.3

# Optional image providers (used for hero images)
PEXELS_API_KEY=...
UNSPLASH_ACCESS_KEY=...

The scripts automatically try multiple Ollama endpoints:
	1.	/v1/chat/completions (OpenAI-compatible)
	2.	/api/chat
	3.	/api/generate
	4.	ollama run ... CLI fallback

⸻

3) education.py — Educational Artifacts from KB + Sources

What it does
	•	Loads your local KB (.md, .txt, .html, .pdf) and optionally scrapes external URLs, sites, or feeds.
	•	Mines topics (or uses your explicit list).
	•	For each topic, generates:
	•	tutorial.md
	•	methodology.md
	•	checklist.md
	•	slides.md (outline)
	•	linkedin.txt
	•	ready.json and manifest.json
	•	images/ (if API keys provided)

All artifacts get YAML front matter for static site tooling.

Quick start

# From local KB only
python3 education.py \
  --kb-dir ./my_kb \
  --language en \
  --topic "Evaluating LLMs in production" \
  --image-count 2 \
  --output-dir output_kb

# Auto-mine topics from feeds and sites
python3 education.py \
  --feeds feeds.yaml \
  --sites sites.yaml \
  --auto-topics 6 \
  --llm-refine-topics \
  --language en \
  --image-count 3 \
  --output-dir output_kb

Inputs
	•	--kb-dir DIR (repeatable) — include local KB folders.
	•	--kb-file FILE (repeatable) — include individual files.
	•	--source-url URL (repeatable) — scrape specific pages.
	•	--sites sites.yaml — crawl start pages and scrape discovered article links.
	•	--feeds feeds.yaml — pull recent links from RSS and scrape them.
	•	--topic "string" / --topics-file file.txt — explicit topics.
	•	--auto-topics N — mine up to N topics from stashed/scraped docs.

Topic mining
	•	Pulls headings (H1/H2/H3) and frequent domain phrases.
	•	Optional --llm-refine-topics to merge near duplicates and keep the top N.

Output layout

output_kb/
  20250906-<topic-slug>-<hash>/
    images/
      img_1.jpg
      img_2.jpg
    tutorial.md
    methodology.md
    checklist.md
    slides.md
    linkedin.txt
    ready.json
    manifest.json

	•	ready.json aggregates all channels and assets for downstream use.
	•	Front matter includes title, slug, date, language, topic, hero_image, sources_used.

Dedup/Registry
	•	output/registry.json prevents regenerating the same topic too frequently.
	•	Keys are normalized; corrupt registries are auto-healed (missing keys are restored).
	•	To re-run everything from scratch, delete output/registry.json.

⸻

4) main.py — News → Multichannel Content

What it does

Scrapes articles or uses free search/RSS to find them, then generates:
	•	post.md (Markdown + front matter + hero image)
	•	twitter_thread.txt (≤ 280 characters per post, formatted/validated)
	•	linkedin.txt (≤ 1,400 characters, formatted/validated)
	•	newsletter.md
	•	manifest.json
	•	images/img_*.jpg

Includes a persistent registry to skip already processed (title+link) articles and a topic tally.

Quick start

# Single URL
python3 main.py \
  --url https://example.com/great-story \
  --language en \
  --image-count 3 \
  --output-dir output

# Crawl defined sites
python3 main.py \
  --sites sites.yaml \
  --language en \
  --max-articles 6

# Free search + GDELT
python3 main.py \
  --use-free-search \
  --q "ai evaluations" \
  --q "zero trust breaches" \
  --language en

# RSS fallback
python3 main.py \
  --feeds feeds.yaml \
  --language en \
  --max-articles 8

Inputs you can mix & match
	•	--url URL — scrape a specific page.
	•	--sites sites.yaml — crawl a site start page and scrape found links.
	•	--use-free-search --q "query" — DuckDuckGo Lite + GDELT to discover links, then scrape.
	•	--feeds feeds.yaml — RSS (topic → list of feeds).
	•	--input-json article.json — pre-built dict: {title, link, source, summary, topic}.

Output layout

output/
  20250906-<title-slug>-<hash>/
    images/
      img_1.jpg
      img_2.jpg
      img_3.jpg
    post.md
    twitter_thread.txt
    linkedin.txt
    newsletter.md
    manifest.json

post.md includes front matter fields like title, slug, date, topic, language, source_url, source_name, hero_image, images[], summary.

Registry behavior
	•	Stored at --registry-path (default: output/registry.json).
	•	De-duplicates by (title + link) within --dedupe-days (default 60).
	•	Maintains topics_tally for quick distribution stats.
	•	Delete the registry file if you want to force reprocessing.

⸻

5) Site & Feed Config Examples

sites.yaml

sites:
  - name: The Verge — AI
    start_url: https://www.theverge.com/ai-artificial-intelligence
    link_selector: "a[href*='/ai-'], a[href*='/ai/']"
    limit: 6
    topic: ai

  - name: ENISA — News
    start_url: https://www.enisa.europa.eu/news
    link_selector: "a[href*='/news/']"
    limit: 6
    topic: cybersecurity

	•	start_url: page to crawl.
	•	link_selector: CSS selector for article links (optional; heuristic fallback otherwise).
	•	limit: max links to pick.
	•	topic: used for routing (ai | blockchain | cybersecurity | anything).

feeds.yaml

topics:
  ai:
    rss:
      - https://ai.googleblog.com/atom.xml
      - https://openai.com/blog/rss.xml
  cybersecurity:
    rss:
      - https://www.bleepingcomputer.com/feed/
      - https://www.darkreading.com/rss.xml

	•	Top-level topics → each key has a rss array of feed URLs.
	•	The scripts parse feeds with feedparser and then scrape the linked pages.

⸻

6) Prompts & Style (What you’ll get)

Both scripts ship with expert, no-fluff prompts:
	•	Clear, short sentences.
	•	No generic blog tropes.
	•	Bulleted lists use - .
	•	Preference for code, commands, configs, and concrete checks.
	•	“Unknown?” → outputs “Not in provided sources.”

education.py produces:
	•	tutorial.md — overview → key concepts → step-by-step → pitfalls → next steps → sources used
	•	methodology.md — purpose → when to use → prerequisites → procedure → acceptance criteria → governance notes → sources used
	•	checklist.md — 10–18 actionable items + notes + sources
	•	slides.md — title, agenda, 10 slides with 3–5 bullets each + sources
	•	linkedin.txt — concise post (validated length)

main.py produces:
	•	post.md — headings: TL;DR → What happened → Why it matters → Context & numbers → Risks & tradeoffs → Actionable next steps → Further reading
	•	twitter_thread.txt — N posts (≤ 280 chars each), with a single CTA at the end
	•	linkedin.txt — hook + short paragraphs + actionable bullets + CTA (validated ≤ 1,400 chars)
	•	newsletter.md — subject options, preheader, hero block, main story, quick hits, further reading, footer note

⸻

7) Images
	•	If PEXELS_API_KEY / UNSPLASH_ACCESS_KEY are set, both tools select a few relevant images.
	•	Images are downloaded into images/ and referenced in front matter as hero_image.
	•	Captions include photographer and provider credits.

Don’t have keys? The tools will simply skip images.

⸻

8) Practical Tips
	•	Model choice: Start with a small, instruction-tuned model in Ollama; bump up only if the content is too terse.
	•	Temperature: Defaults are conservative (0.3–0.4). Increase slightly for more variation.
	•	PDFs: If parsing is slow or noisy, remove PDFs from your KB or set a separate KB folder for .md only.
	•	Scraping: Some sites block bots. If a page fails, the script just skips it.
	•	Length: main.py uses --min-words/--max-words to size the blog; social posts are validated post-generation.
	•	Registry: If you see “Skipping (already covered recently)”, either change input or delete output/registry.json.

⸻

9) Troubleshooting
	•	KeyError: 'tally' (older registry format):
	•	Fixed in education.py: the registry auto-heals missing keys. Delete output/registry.json if you still see issues.
	•	pdfminer.six not installed, skipping PDF:
	•	Install pdfminer.six or ignore if you don’t need PDFs.
	•	Ollama endpoint errors:
	•	Check OLLAMA_BASE, confirm model is pulled: ollama pull llama3.1:8b-instruct-q4_K_M.
	•	The scripts try multiple API paths + CLI fallback.
	•	Empty outputs:
	•	Ensure your KB has non-trivial text or your sites/feeds actually returned pages with content.
	•	Use --index-out output/index.json (education.py) to see what was indexed.

⸻

10) Example Commands (Copy/Paste)

# Build educational packs for mined topics from RSS + sites
python3 education.py \
  --feeds feeds.yaml \
  --sites sites.yaml \
  --auto-topics 6 \
  --llm-refine-topics \
  --language en \
  --image-count 2 \
  --output-dir output_kb

# Turn a scraped article into a multichannel bundle
python3 main.py \
  --url https://example.com/great-story \
  --language en \
  --image-count 3 \
  --output-dir output


⸻

11) What to put in your KB
	•	Concise .md notes from docs, runbooks, incident reviews, and code comments.
	•	Policy/config snippets and their rationale.
	•	“Gotchas” and post-deployment checks.
	•	Avoid: marketing PDFs, low-signal slides, overly long transcripts.

⸻

12) License & Contributions
	•	MIT-like project layout (pick your license).
	•	PRs welcome for better scraping heuristics, model adapters, or new output channels.

Happy shipping.

