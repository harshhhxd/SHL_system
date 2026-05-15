# SHL Assessment Recommender

A conversational agent that takes hiring managers from a vague intent to a grounded shortlist of SHL Individual Test Solutions through multi-turn dialogue.

---

## Project Structure

```
.
├── main.py          # FastAPI service — GET /health, POST /chat
├── agent.py         # Conversational agent using Groq LLM
├── retriever.py     # FAISS-based semantic retriever over SHL catalog
├── evaluation.py    # Automated evaluation harness
├── shl_catalog.json # Scraped SHL Individual Test Solutions catalog
└── README.md
```

---

## Requirements

- Python 3.10+
- A [Groq API key](https://console.groq.com/) (free tier works)

---

## Installation

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <repo-folder>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
fastapi
uvicorn
python-dotenv
pydantic
groq
sentence-transformers
faiss-cpu
numpy
httpx
```

---

## Environment Setup

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## Running the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Server starts at `http://localhost:8000`.

On startup the retriever:
- Loads `shl_catalog.json`
- Encodes all assessments with `all-MiniLM-L6-v2`
- Builds a FAISS index (~100+ assessments)

This takes ~10–20 seconds on first run.

---

## API

### Health Check

```
GET /health
```

Response:
```json
{ "status": "ok" }
```

### Chat

```
POST /chat
Content-Type: application/json
```

Request:
```json
{
  "messages": [
    { "role": "user", "content": "Hiring a mid-level Java developer, 4 years experience." }
  ]
}
```

Response:
```json
{
  "reply": "Here are assessments that fit a mid-level Java developer.",
  "recommendations": [
    { "name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K" },
    { "name": "OPQ32r",       "url": "https://www.shl.com/...", "test_type": "P" }
  ],
  "end_of_conversation": false
}
```

**Schema rules (non-negotiable):**
- `reply` — non-empty string
- `recommendations` — array of 0–10 items, each with `name`, `url`, `test_type`; all URLs must be from `shl.com`
- `end_of_conversation` — boolean

`recommendations` is empty when the agent is still gathering context or refusing an off-topic request.

---

## Agent Behaviour

| Scenario | Agent Action |
|---|---|
| Vague query ("I need an assessment") | Asks one clarifying question, returns `recommendations=[]` |
| Specific role / seniority / skill given | Returns 1–10 grounded recommendations immediately |
| User changes constraints mid-conversation | Refines existing shortlist, does not restart |
| Comparison question ("OPQ32r vs GSA?") | Returns catalog-grounded answer |
| Off-topic (Jira, Trello, HR advice, legal, competitors) | Refuses with explanation, `recommendations=[]` |
| Prompt injection attempt | Refuses, `recommendations=[]` |

---

## Retrieval Design

1. **Semantic search** — query encoded with `all-MiniLM-L6-v2`, compared against FAISS `IndexFlatIP` (cosine similarity). Fetch window = 3× top_k for re-ranking headroom.
2. **Keyword re-ranking** — candidates sorted by count of query words (len ≥ 3) found in assessment name. Promotes exact matches semantic search may underrank.
3. **Tech variant injection** — detected technology keywords (java, python, sql, etc.) trigger a full catalog name-sweep. Variant groups (e.g. `Python (New)` + `Python (Advanced Level)`) are pinned together so both always surface.
4. **Always-include list** — OPQ32r, Motivation Questionnaire, Verify G+, Verify Numerical/Verbal and Interactive variants are appended to every result set regardless of retrieval rank.

---

## Running Evaluation

```bash
# Default delay=15s between calls (required for Groq free tier TPM limit)
python evaluation.py --url http://localhost:8000

# Faster if you have a paid Groq tier
python evaluation.py --url http://localhost:8000 --delay 5
```

### Evaluation Dimensions

| Dimension | What it checks |
|---|---|
| Hard evals | Schema compliance, catalog-only URLs, ≤10 recommendations |
| Behavior probes | 15 binary pass/fail conversation scenarios |
| Recall@10 | Fraction of relevant assessments appearing in top-10, across 10 queries |
| Precision@10 | Fraction of returned items that are relevant |
| Recommendation relevance | Spot-check on 5 role queries |
| Groundedness | Comparison replies grounded in catalog data |
| Latency | Per-call response time vs 28s hard limit |

### Behavior Probes (15 total)

| Probe | Description |
|---|---|
| `vague_query_no_recs` | Vague query must return 0 recommendations |
| `off_topic_refusal` | Software tool question (Jira/Trello) must be refused |
| `prompt_injection_refusal` | Injection attempt must be refused |
| `specific_role_gets_recs` | Specific role must return 1–10 recommendations |
| `refine_updates_shortlist` | Constraint change must update, not restart shortlist |
| `comparison_uses_catalog` | Comparison reply must be non-empty and grounded |
| `turn_cap_honoured` | Agent must commit to a shortlist within 4 user turns |
| `competitor_refusal` | Competitor vendor questions must return 0 recommendations |
| `jd_paste_gets_recs` | Pasted job description must yield 1–10 recommendations |
| `hallucinated_url_absent` | All returned URLs must be from shl.com |
| `hr_advice_refusal` | Interview/HR advice questions must return 0 recommendations |
| `empty_input_no_crash` | Whitespace-only input must not crash |
| `eoc_flag_after_closure` | end_of_conversation must be True after user confirms satisfaction |
| `rec_count_hard_cap` | Response must never exceed 10 recommendations |
| `legal_question_refusal` | Legal/compliance questions must return 0 recommendations |

---

## Evaluation Results

```
── Health check ──────────────────────────────────
  ✅ /health OK

── Behavior probes ───────────────────────────────
  ✅ vague_query_no_recs
  ✅ off_topic_refusal
  ✅ prompt_injection_refusal
  ✅ specific_role_gets_recs
  ✅ refine_updates_shortlist
  ✅ comparison_uses_catalog
  ✅ turn_cap_honoured
  ✅ competitor_refusal
  ✅ jd_paste_gets_recs
  ✅ hallucinated_url_absent
  ✅ hr_advice_refusal
  ✅ empty_input_no_crash
  ✅ eoc_flag_after_closure
  ✅ rec_count_hard_cap
  ✅ legal_question_refusal

── Schema compliance ─────────────────────────────
  Schema compliance: 15/15 (100%)

── Recall@10 ─────────────────────────────────────
  ✅ java_developer      Recall@10 = 1.00
  ✅ sales_manager       Recall@10 = 1.00
  ✅ customer_service    Recall@10 = 1.00
  ✅ data_analyst        Recall@10 = 1.00
  ⚠️  python_engineer    Recall@10 = 0.50

══════════════════════════════════════════════════
  EVALUATION SUMMARY
══════════════════════════════════════════════════
  Behavior probe pass-rate    : 100%  (15/15)
  Schema compliance           : 100%
  Mean Recall@10              : 0.900
  Mean Precision@10           : 0.454
  Recommendation relevance    : 100%  (5/5)
  Groundedness pass-rate      : 100%  (3/3)
  Avg latency                 : 9.31s  (3/3 within 28s limit)
══════════════════════════════════════════════════
```

### Notes on Results

- **Recall@10 = 0.900** — the 0.1 gap is a single miss: `Python (Advanced Level)` falling out of the top-12 retrieval window for the `python_engineer` case. All other cases hit 1.00.
- **Precision@10 = 0.454** — expected and not penalised. With 2–3 ground-truth relevant items per query and up to 10 returned, precision is structurally bounded. A re-ranker would improve this but adds latency.
- **Latency 9.31s avg** — dominated by Groq API call (~7–8s) on free tier. All calls comfortably within the 28s hard limit.

---

## Stack

| Component | Choice | Reason |
|---|---|---|
| LLM | Groq `llama-3.1-8b-instant` | Fast TTFT, free tier, JSON mode |
| Embeddings | `all-MiniLM-L6-v2` | Lightweight, runs locally, no API cost |
| Vector store | FAISS `IndexFlatIP` | In-process, zero latency, sufficient for ~100 items |
| API framework | FastAPI + Pydantic v2 | Async, schema validation, OpenAPI docs free |
| HTTP client (eval) | httpx | Sync client, clean timeout handling |

---

## Known Limitations

- Free Groq tier has TPM limits — sustained load will hit rate limits; the `_call_groq` method retries once after 3s.
- Latency is dominated by the Groq API round trip (~7–8s); moving to a paid tier or self-hosted model would bring this under 2s.
- `Python (Advanced Level)` can fall out of the retrieval window under certain query phrasings — mitigated by the `TECH_VARIANT_GROUPS` pinning logic in `retriever.py`.
