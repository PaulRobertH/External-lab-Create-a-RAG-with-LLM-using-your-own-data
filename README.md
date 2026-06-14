# Wine Portfolio RAG Assistant 🍷

An implementation of the **Retrieval Augmented Generation (RAG)** pattern over my own data —
a real Nexus wine portfolio export — built for the *Introduction to Retrieval Augmented
Generation* practice lab. It replaces the example wine-review dataset with my actual cellar stock.

Ask a natural question like *"Which Italian reds do I have, and are any ready to drink?"* and the
app retrieves the most relevant wines from a vector database and has a local LLM write a grounded
answer using only those wines.

## How it works

```
CSV  ──Pandas──>  list of dicts  ──Sentence Transformers──>  embeddings
                                                                 │
                                                          Qdrant (in-memory)
                                                                 │
                              question ──embed──> semantic search (top-k)
                                                                 │
                                   retrieved context + question ─┴─> LLM (Llamafile) ─> answer
```

| Stage | Tool |
|-------|------|
| Data loading | Pandas |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector database | Qdrant (runs in-memory, nothing to host) |
| LLM | Llamafile (OpenAI-compatible) — or any OpenAI endpoint |
| LLM client | OpenAI Python SDK |

## The data

`data/wine_portfolio.csv` — 112 wine holdings, each with `name`, `producer`, `type`, `country`,
`region`, `vintage`, `bottles`, `status`, `storage`, `duty_status`, `drink_window`, `roi_pct`,
`case_value_gbp`, and a free-text `description`. The **description** column is what gets embedded;
the whole row is stored as the payload so structured fields come back with each result.

This was generated from a raw portfolio export by combining the meaningful columns into one
descriptive sentence per wine. To use different data, replace this CSV (keep one text column to
embed) and update the column name in the code.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download a Llamafile LLM (the Phi-2 model, ~2 GB, works well)
#    See https://github.com/Mozilla-Ocho/llamafile#other-example-llamafiles
chmod +x phi-2.Q4_K_M.llamafile

# 3. Start the LLM server (OpenAI-compatible, listens on :8080)
./phi-2.Q4_K_M.llamafile --server --nobrowser
```

> **No Llamafile?** Point the client at any OpenAI-compatible endpoint instead by editing
> `LLM_BASE_URL` (and the API key) in `rag.py`, or `base_url`/`api_key` in the notebook.

## Run it

**Option A — Jupyter notebook** (recommended, step by step):

```bash
jupyter notebook embeddings.ipynb
```

**Option B — Standalone Python app:**

```bash
python rag.py                                   # interactive chat
python rag.py "what are my best wines by ROI?"  # one-off question
```

## Example questions to try

- "Which Italian reds do I have, and are any ready to drink?"
- "What are my best wines by ROI?"
- "Which white wines are in stock?"
- "What is stored in bond at Nexus?"

## Learning objectives covered

- ✅ Implement the RAG pattern with your own data
- ✅ Apply your own data (a real wine portfolio) to solve a problem using RAG
- ✅ Leverage an LLM + a vector database (Qdrant) for useful, grounded responses
- ✅ Create embeddings with Sentence Transformers
- ✅ Use the OpenAI Python API to connect to a local LLM (Llamafile)
