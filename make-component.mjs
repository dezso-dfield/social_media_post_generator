// tools/generate-from-output.mjs
// Run: GEN_VARIANTS=post node tools/generate-from-output.mjs
// Purpose: 1 React component per *selected* variant per bundle (default only "post").
// - Scans output_kb/ and output/
// - Prefers output_kb, else newest mtime
// - Strong de-dupe by normalized title (channel stripped) + variant
// - Copies images next to component
// - Writes dynamic routes for vite-react-ssg using <Head />
// - Adds prefixed classnames (blogpost-*) and imports Blog.css
// - Renders a display date with a random time BEFORE the published time (same day)

import fs from 'node:fs/promises'
import path from 'node:path'
import { createHash } from 'node:crypto'
import matter from 'gray-matter'
import prettier from 'prettier'

/* ───────────── config ───────────── */
const ROOT = process.cwd()
const SCAN_ROOTS = [path.join(ROOT, 'output_kb'), path.join(ROOT, 'output')]
const GEN_ROOT = path.join(ROOT, 'src/components/GeneratedPosts')

// Which variants to generate. Default: only "post".
// Accepts comma list env: GEN_VARIANTS=post,methodology
const ALL_KNOWN = ['post', 'methodology', 'checklist', 'tutorial', 'slides', 'newsletter']
const WANT = (process.env.GEN_VARIANTS || 'post')
  .split(',')
  .map(s => s.trim().toLowerCase())
  .filter(s => ALL_KNOWN.includes(s))
const SELECTED = WANT.length ? WANT : ['post']

/* ───────────── fs utils ───────────── */
const exists = async p => !!(await fs.stat(p).catch(() => null))
const ensureDir = d => fs.mkdir(d, { recursive: true })
const rmrf = async d => (await exists(d)) && fs.rm(d, { recursive: true, force: true })
const fileMtime = async p => {
  try { const st = await fs.stat(p); return Number(st.mtimeMs || 0) } catch { return 0 }
}

/* ───────────── string / hash helpers ───────────── */
const sha = s => createHash('sha1').update(String(s)).digest('hex')
const sh6 = s => sha(s).slice(0, 6)

const slugify = s =>
  (s || '')
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/[\s_-]+/g, '-')
    .replace(/^-+|-+$/g, '')

const normTitle = (t = '') =>
  String(t)
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .replace(/[^\p{L}\p{N}\s-]+/gu, '')
    .trim()

// strip channel suffix/prefix like "— Checklist", "[Tutorial] "
const CHANNELS = ALL_KNOWN
const stripChannelFromTitle = (t = '') => {
  let s = String(t).replace(/\s+/g, ' ').trim()
  const join = CHANNELS.join('|')
  const suffixRe = new RegExp(`(?:\\s*[—:-]\\s*)\\b(?:${join})\\b\\s*$`, 'i')
  s = s.replace(suffixRe, '').trim()
  const bracketRe = new RegExp(`\\s*[\$begin:math:display$(]\\\\s*(?:${join})\\\\s*[\\$end:math:display$)]\\s*$`, 'i')
  s = s.replace(bracketRe, '').trim()
  const prefixRe = new RegExp(`^\\b(?:${join})\\b\\s*[:—-]\\s*`, 'i')
  s = s.replace(prefixRe, '').trim()
  return s
}

/* ───────────── tiny MD → HTML ───────────── */
function mdToHtml(md) {
  let s = md.replace(/</g, '&lt;').replace(/>/g, '&gt;')
  s = s.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_m, alt, src) => `<img alt="${alt}" src="${src}" />`)
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_m, text, url) => `<a href="${url}" target="_blank" rel="noopener noreferrer">${text}</a>`)
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  s = s.replace(/\*([^*]+)\*/g, '<em>$1</em>')
  s = s.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>')
  s = s.replace(/^##\s+(.+)$/gm, '<h2>$1</h2>')
  s = s.replace(/^#\s+(.+)$/gm, '<h1>$1</h1>')
  s = s.replace(/^\s*-\s+(.*)$/gm, '<li>$1</li>')
  s = s.replace(/((?:<li>[\s\S]*?<\/li>\s*)+)/g, m => `<ul>${m}</ul>`)
  s = s
    .split(/\n{2,}/)
    .map(bl => {
      const t = bl.trim()
      return t.startsWith('<') ? t : (t ? `<p>${t}</p>` : '')
    })
    .filter(Boolean)
    .join('\n')
  return s
}

/* ───────────── filename → variant ───────────── */
function mapVariant(basename) {
  const n = basename.toLowerCase()
  if (n === 'post.md') return 'post'
  if (n === 'methodology.md') return 'methodology'
  if (n === 'checklist.md') return 'checklist'
  if (n === 'tutorial.md') return 'tutorial'
  if (n === 'slides.md') return 'slides'
  if (n === 'newsletter.md') return 'newsletter'
  return null
}

/* ───────────── misc helpers ───────────── */
async function copyDir(src, dst) {
  if (!(await exists(src))) return
  await ensureDir(dst)
  const ents = await fs.readdir(src, { withFileTypes: true })
  for (const e of ents) {
    const s = path.join(src, e.name)
    const d = path.join(dst, e.name)
    if (e.isDirectory()) await copyDir(s, d)
    else if (e.isFile()) await fs.copyFile(s, d).catch(() => null)
  }
}

async function* walk(dir) {
  const stack = [dir]
  while (stack.length) {
    const cur = stack.pop()
    let ents
    try {
      ents = await fs.readdir(cur, { withFileTypes: true })
    } catch {
      continue
    }
    let hasMd = false
    for (const e of ents) {
      if (e.isDirectory()) stack.push(path.join(cur, e.name))
      else if (e.isFile() && e.name.toLowerCase().endsWith('.md')) hasMd = true
    }
    if (hasMd) yield cur
  }
}

const plain = (html = '') => String(html).replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim()
const clip = (s, n = 300) => (s.length <= n ? s : s.slice(0, n - 1) + '…')

/* ───────────── pick a display date (random time before published within the day) ───────────── */
function pickDisplayDateISO(publishedISO) {
  const pub = new Date(publishedISO || Date.now())
  if (Number.isNaN(pub.getTime())) return new Date().toISOString()

  const start = new Date(pub.getFullYear(), pub.getMonth(), pub.getDate(), 0, 0, 0, 0)
  let windowMs = pub.getTime() - start.getTime() - 60 * 1000 // leave at least 1 minute gap
  if (windowMs < 0) windowMs = 0
  const randOffset = Math.floor(Math.random() * (windowMs + 1))
  const display = new Date(start.getTime() + randOffset)
  return display.toISOString()
}

/* ───────────── component template ───────────── */
function buildComponentCode({ front, html, heroLocalImport, routePath }) {
  const title = stripChannelFromTitle(front.title || 'Untitled')
  const lang = front.language || 'en'
  const desc = clip(front.summary || plain(html), 250)
  const published = front.date || new Date().toISOString()
  const displayISO = pickDisplayDateISO(published) // <= random earlier time same day
  const tags = Array.isArray(front.tags) ? front.tags : []
  const sources = Array.isArray(front.sources_used) ? front.sources_used : []

  const metaTitle = JSON.stringify(title)
  const metaDesc = JSON.stringify(desc)
  const metaDate = JSON.stringify(published)
  const displayDateISO = JSON.stringify(displayISO)

  const heroFig = heroLocalImport
    ? `
        <figure className="blogpost-hero">
          <img src={hero} alt=${JSON.stringify(title)} loading="lazy" />
        </figure>`
    : ''

  return `import React from 'react'
import { Head } from 'vite-react-ssg'
import '@/components/Blog/Blog.css'
${heroLocalImport ? `const hero = new URL('${heroLocalImport}', import.meta.url).href` : ''}

export const meta = ${JSON.stringify(
    {
      title,
      slug: front.slug,
      language: lang,
      date: published,
      topic: front.topic || '',
      channel: front.__channel || '',
      tags,
      sources_used: sources,
      route: routePath,
      displayDateISO, // human-friendly date/time we show in the UI
    },
    null,
    2
  )}

const Page = () => {
  const displayDate = new Date(${displayDateISO}).toLocaleString(${JSON.stringify(lang)}, { year:'numeric', month:'short', day:'2-digit', hour:'2-digit', minute:'2-digit' })
  const publishedDate = new Date(${metaDate}).toISOString()

  return (
    <>
      <Head>
        <html lang=${JSON.stringify(lang)} />
        <title>{${metaTitle}}</title>
        <meta name="description" content={${metaDesc}} />
        <meta property="og:type" content="article" />
        <meta property="og:title" content={${metaTitle}} />
        <meta property="og:description" content={${metaDesc}} />
        ${heroLocalImport ? `<meta property="og:image" content={hero} />` : ''}
        <meta property="article:published_time" content={publishedDate} />
        ${tags.map(t => `<meta property="article:tag" content=${JSON.stringify(String(t))} />`).join('\n        ')}
      </Head>

      <article className="blogpost-article">
        <header className="blogpost-header">
          <h1 className="blogpost-title">{${metaTitle}}</h1>
          <p className="blogpost-meta">
            <time className="blogpost-date" dateTime={publishedDate}>{displayDate}</time>
            {${JSON.stringify(lang.toUpperCase())}}
          </p>
          ${heroFig}
        </header>

        <section className="blogpost-body" dangerouslySetInnerHTML={{ __html: ${JSON.stringify(html)} }} />

        ${
          sources.length
            ? `<section className="blogpost-sources">
                 <h3 className="blogpost-sources-title">Sources used</h3>
                 <ul className="blogpost-sources-list">${sources
                   .map(s => `<li class="blogpost-source-item">- ${String(s).replace(/</g, '&lt;').replace(/>/g, '&gt;')}</li>`)
                   .join('')}</ul>
               </section>`
            : ''
        }
      </article>
    </>
  )
}

export default Page
`
}

/* ───────────── convert one MD ───────────── */
async function processOne({ mdPath, bundleDir, variant }) {
  const raw = await fs.readFile(mdPath, 'utf-8')
  const { data: front, content } = matter(raw)

  const folderBase = path.basename(bundleDir)
  const baseSlug = `${slugify(folderBase)}-${sh6(bundleDir)}`
  const finalSlug = `${baseSlug}-${variant}`

  const compDir = path.join(GEN_ROOT, finalSlug)
  await rmrf(compDir)
  await ensureDir(compDir)

  // copy images → ./assets
  const srcImages = path.join(bundleDir, 'images')
  const dstAssets = path.join(compDir, 'assets')
  await copyDir(srcImages, dstAssets)

  // hero import if available
  let heroLocalImport = null
  const fmHero = front.hero_image ? path.basename(front.hero_image) : null
  if (fmHero && (await exists(path.join(dstAssets, fmHero)))) {
    heroLocalImport = `./assets/${fmHero}`
  } else if (await exists(path.join(dstAssets, 'img_1.jpg'))) {
    heroLocalImport = './assets/img_1.jpg'
  }

  const html = mdToHtml(content)
  front.slug = finalSlug
  front.__channel = variant

  const routePath = `/en/blog/${finalSlug}`
  const code = buildComponentCode({ front, html, heroLocalImport, routePath })
  const pretty = await prettier.format(code, { parser: 'babel' })
  await fs.writeFile(path.join(compDir, 'index.jsx'), pretty, 'utf-8')

  return { slug: finalSlug, route: routePath, title: front.title || folderBase, channel: variant }
}

/* ───────────── main ───────────── */
async function main() {
  // Clean generated target every run
  await rmrf(GEN_ROOT)
  await ensureDir(GEN_ROOT)

  const routes = []

  // 1) discover bundle dirs
  const found = []
  for (const root of SCAN_ROOTS) {
    if (!(await exists(root))) continue
    for await (const dir of walk(root)) found.push(dir)
  }
  console.log(`🧭 Found ${found.length} bundle folder(s).`)

  // 2) collapse by folder name (prefer output_kb, else newest mtime)
  const byName = new Map()
  for (const dir of found) {
    const base = path.basename(dir)
    if (!byName.has(base)) { byName.set(base, dir); continue }
    const prev = byName.get(base)

    const bestTime = async d => {
      const names = await fs.readdir(d).catch(() => [])
      const md = names.filter(n => n.toLowerCase().endsWith('.md'))
      const times = await Promise.all(md.map(n => fileMtime(path.join(d, n))))
      return Math.max(0, ...times)
    }

    const preferKb = dir.includes('output_kb') && !prev.includes('output_kb')
    if (preferKb) byName.set(base, dir)
    else {
      const [tNew, tPrev] = await Promise.all([bestTime(dir), bestTime(prev)])
      if (tNew > tPrev) byName.set(base, dir)
    }
  }
  const bundleDirs = Array.from(byName.values())

  // 3) for each bundle: pick only SELECTED variants; de-dupe by normalized title/content + variant
  const bestByKey = new Map()

  for (const bundle of bundleDirs) {
    const names = await fs.readdir(bundle)
    const mdFiles = names.filter(n => n.toLowerCase().endsWith('.md'))

    const pairs = mdFiles
      .map(n => ({ name: n, variant: mapVariant(n) }))
      .filter(x => x.variant && SELECTED.includes(x.variant))

    if (!pairs.length && mdFiles[0] && SELECTED.includes('post')) {
      pairs.push({ name: mdFiles[0], variant: 'post' })
    }

    for (const { name, variant } of pairs) {
      const mdPath = path.join(bundle, name)

      let titleKey = ''
      let contentKey = ''
      try {
        const raw = await fs.readFile(mdPath, 'utf-8')
        const { data, content } = matter(raw)
        const baseTitle = stripChannelFromTitle(data?.title || '')
        titleKey = normTitle(baseTitle)
        contentKey = sha(content).slice(0, 12)
      } catch { continue }

      const key = `${titleKey || `hash:${contentKey}`}|${variant}`
      const mtime = await fileMtime(mdPath)
      const prev = bestByKey.get(key)
      const preferThis =
        !prev ||
        mtime > prev.mtime ||
        (mtime === prev.mtime && bundle.includes('output_kb') && !prev.bundleDir.includes('output_kb'))

      if (preferThis) bestByKey.set(key, { mdPath, bundleDir: bundle, variant, mtime })
    }
  }

  // 4) generate once per key
  for (const item of bestByKey.values()) {
    const res = await processOne(item)
    if (res) routes.push(res)
  }

  // 5) write routes + manifest
  const uniqByPath = new Map()
  for (const r of routes) if (!uniqByPath.has(r.route)) uniqByPath.set(r.route, r)
  const finalRoutes = Array.from(uniqByPath.values())

  const routesFile = `// AUTO-GENERATED — do not edit.
import React, { Suspense, lazy } from 'react'
const wrap = (loader) => {
  const C = lazy(loader)
  return (
    <Suspense fallback={<div style={{padding:'2rem'}}>Loading…</div>}>
      <C />
    </Suspense>
  )
}
export const generatedBlogRoutes = [
${finalRoutes.map(r => `  { path: '${r.route}', element: wrap(() => import('./${r.slug}/index.jsx')) }`).join(',\n')}
]
`
  await fs.writeFile(path.join(GEN_ROOT, '_generated.routes.jsx'), routesFile, 'utf-8')
  await fs.writeFile(path.join(GEN_ROOT, '_generated.manifest.json'), JSON.stringify(finalRoutes, null, 2), 'utf-8')

  console.log(`\n✅ Generated ${finalRoutes.length} component(s) for variant(s): [${SELECTED.join(', ')}]`)
  const sample = finalRoutes.slice(0, 6).map(r => `• ${r.title} → ${r.route}`).join('\n')
  if (sample) console.log(sample)
  console.log(`\n→ Import routes:\n  import { generatedBlogRoutes } from '@/components/GeneratedPosts/_generated.routes.jsx'`)
}

main().catch(err => { console.error('❌ Generator failed:', err); process.exit(1) })