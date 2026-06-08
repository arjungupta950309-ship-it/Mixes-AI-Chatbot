---
title: King Arthur Mixes Assistant
emoji: 🧁
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
short_description: AI assistant for King Arthur Baking's Mixes catalog (RAG + LangGraph)
---

# 🧁 King Arthur Baking — Mixes Assistant

An AI chatbot that answers questions about the **Mixes** category of
[King Arthur Baking](https://shop.kingarthurbaking.com/mixes) — flavors, prices,
dietary options (gluten-free, kosher…), and recommendations — grounded in data
scraped directly from the store.

Built with **LangChain + LangGraph** (a self-correcting RAG agent), **Chroma**
for vector search, and a **Streamlit** frontend, deployed on **Hugging Face
Spaces**.

---

## How it works

```
shop.kingarthurbaking.com/mixes
        │  (1) scraper/scrape.py  — requests + BeautifulSoup
        ▼
   data/products.json   (118 products)
        │  (2) ingest.py  — chunk + embed
        ▼
   chroma_db/           (persistent vector store)
        │  (3) agent.py  — LangGraph agent
        ▼
   Streamlit app.py     — chat + live agent-graph plot
```

### The agent (LangGraph)

This is an **agentic** retrieval pipeline, not a plain retrieve-then-answer
chain. It self-checks retrieval quality and reformulates weak queries before
answering:

```
retrieve → grade → (relevant?) ── yes ──► generate → END
                       │
                       no, retries left
                       ▼
                   rewrite → retrieve (loop)
```

- **retrieve** — semantic search over the product catalog
- **grade** — the LLM judges whether the retrieved products are relevant
- **rewrite** — if not, reformulate the query and retry (up to 2×)
- **generate** — answer grounded *only* in the catalog, with sources

The compiled graph is rendered live in the app's sidebar.

---

## Providers (all free, configurable via env)

The app is **provider-flexible** — pick a backend with two env vars. The default
runs a small model **on-device** (no API key, works in any region):

| Knob | Default | Other options |
|---|---|---|
| `LLM_PROVIDER` | `local` (Qwen2.5-0.5B-Instruct) | `huggingface`, `gemini`, `github`, `openai`, `groq` |
| `EMBED_PROVIDER` | `huggingface` (all-MiniLM-L6-v2, local) | `openai` |

See [.env.example](.env.example) for every variable.

---

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env          # defaults work with no API key (local model)

# (optional) re-scrape + rebuild the knowledge base
python scraper/scrape.py
python ingest.py

streamlit run app.py
```

The first run downloads the small local model (~1 GB) once.

---

## Deploy to Hugging Face Spaces

1. Create a new **Space** → SDK **Streamlit**.
2. Upload this repo (including `data/products.json` and `chroma_db/` so the Space
   doesn't need to re-scrape or re-embed).
3. If you use a hosted provider (e.g. `huggingface`/`openai`), add the key as a
   **Space secret** (`HF_TOKEN`, `OPENAI_API_KEY`, …). The default `local`
   provider needs **no secret**.
4. The Space builds from `requirements.txt` and launches `app.py` automatically.

---

## Project layout

```
king-mixes-bot/
├── app.py              # Streamlit UI + agent-graph plot
├── agent.py            # LangGraph self-correcting RAG agent
├── ingest.py           # build the Chroma vector store
├── providers.py        # swappable LLM / embedding providers
├── config.py           # paths, provider selection, model names
├── scraper/scrape.py   # King Arthur "Mixes" scraper
├── data/products.json  # scraped knowledge base (118 products)
├── chroma_db/          # persisted vector store
├── requirements.txt
└── README.md
```
