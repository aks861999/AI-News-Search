"""Yahoo Finance → LLM → A2UI backend.

Flow:
  1. User asks a finance question (POST /api/ask).
  2. Live Yahoo results fetched via DDGS (strict backend, DNS retry logic).
  3. NVIDIA NIM Nemotron synthesizes the results into Google A2UI v1.0 JSON.
  4. Frontend (React / Lit / Flutter A2UI renderer) renders the payload.

A2UI v1.0 wire format (verified against the spec source in ai-resources):
  [ {"version": "1.0",
     "createSurface": {"surfaceId": ..., "catalogId": ...<basic catalog URL>}},
    {"version": "1.0",
     "updateComponents": {"surfaceId": ..., "components": [...]}} ]

Standard catalog components (NO MetricTile — use Card/Row/Text):
  Card (single `child`), Column/Row/List (`children`: array of IDs),
  Text (`text`: literal string or {"@path": ...}), Image, Icon, Divider,
  Button, TextField, CheckBox, Tabs, Modal.
"""

import json
import logging
import os
import time
from typing import Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException, TimeoutException
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, Field

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("finance-a2ui")

app = FastAPI(title="Yahoo Finance A2UI Agent")

# Allow your React / Lit / Flutter A2UI frontend to call this API.
# Narrow allow_origins to your frontend's URL before public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. NVIDIA NIM client ---
# Real endpoint (the docs' base URL is https://integrate.api.nvidia.com, /v1 for OpenAI SDK):
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    log.warning("NVIDIA_API_KEY is not set! Set it before starting the server.")

client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY or "missing")

# Model IDs on NVIDIA NIM (Build.NVIDIA.com):
#   nvidia/nemotron-3.5-super-vl-120b-a12b  <- top pick: 120B/12B-active MoE vision-language,
#       tool use + structured outputs, 1M context, native MTP speculative decoding
#   nvidia/nemotron-3-ultra-550b-a55b       <- fallback (bigger, stronger)
#   nvidia/nemotron-3.5-bolt-e4b            <- fastest, for low-latency streaming
# If the API returns "model not found", check the exact id on the model's NIM page
# (card name NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 -> lowercase dash id on NIM).
MODEL_ID = os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")


# --- 2. DDGS search locked to Yahoo (same retry logic as app.py) ---
def search_yahoo(
    query: str,
    category: str = "Web (text)",
    region: str = "us-en",
    timelimit: str | None = "m",
    max_results: int = 8,
    proxy: str | None = None,
    retries: int = 3,
) -> list[dict[str, Any]]:
    """Query Yahoo via DDGS with transient-DNS retry logic."""
    backend = "yahoo"
    ddgs = DDGS(proxy=proxy or None, timeout=15)
    kwargs = dict(
        query=query,
        region=region,
        timelimit=timelimit,
        max_results=max_results,
        backend=backend,
    )
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            results = ddgs.news(**kwargs) if category == "News" else ddgs.text(**kwargs)
            for r in results:
                r["_backend"] = backend
            log.info("backend=%s query=%r -> %d results", backend, query, len(results))
            return results
        except (DDGSException, TimeoutException) as e:
            last_exc = e
            is_dns = "DNSError" in str(e) or "Query Refused" in str(e)
            if attempt < retries and is_dns:
                log.warning("Transient DNS error on attempt %d/%d — retrying...", attempt, retries)
                time.sleep(1.0 * attempt)
            elif attempt < retries:
                raise
    raise last_exc  # type: ignore[misc]


# --- 3. A2UI v1.0 system prompt (matches the real spec) ---
A2UI_SYSTEM_PROMPT = """You are a financial analyst agent. You output ONLY Google A2UI v1.0 JSON messages.

You must return a raw JSON ARRAY of exactly two message objects (no markdown fences, no prose):

[
  {
    "version": "1.0",
    "createSurface": {
      "surfaceId": "finance",
      "catalogId": "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json",
      "sendDataModel": true
    }
  },
  {
    "version": "1.0",
    "updateComponents": {
      "surfaceId": "finance",
      "components": [ ...component list... ]
    }
  }
]

Component rules (A2UI v1.0 Basic Catalog — these are the ONLY components):
- Card: {"id": "...", "component": "Card", "child": "<single child component id>"}
  Card takes ONE child. To put multiple things in a card, wrap them in a Column/Row and
  pass that container's id as `child`.
- Column / Row / List: {"id": "...", "component": "Column", "children": ["<id>", "<id>", ...]}
  `children` is an ARRAY OF COMPONENT IDS. Components CANNOT be nested inline — every child
  must be its own entry in the components list, referenced by id.
- Text: {"id": "...", "component": "Text", "text": "<literal string>", "variant": "body"|"caption"}
  Use variant "caption" for smaller source lines.
- Icon: {"id": "...", "component": "Icon", "name": "<material icon name>"}
- Divider: {"id": "...", "component": "Divider"}

Layout guidance for a finance answer:
- root: a Card whose child is a Column.
- The Column holds: a header Row (Icon + title Text), a metrics Row (2-4 Text "chips"),
  a summary Text (2-4 sentences), a Divider, and a sources Text (variant "caption")
  listing the source URLs from the search results.
- Put key numbers (price, growth %, revenue) as short bold-markdown Text lines in the
  metrics Row. There is no MetricTile component — use Text.

Grounding rules:
1. Ground every number and claim strictly in the provided search results. Never invent data.
2. Every component id referenced in `child`/`children` MUST exist as a component entry.
3. Return ONLY the raw JSON array. No markdown fences, no explanation.
"""


# --- 4. API models ---
class FinanceQueryRequest(BaseModel):
    question: str
    category: str = Field(default="Web (text)", pattern=r"^(Web \(text\)|News)$")
    timelimit: str | None = "m"
    max_results: int = Field(default=8, ge=1, le=25)


@app.post("/api/ask")
def ask_finance(req: FinanceQueryRequest):
    # Step A: fetch live Yahoo results via DDGS
    try:
        search_query = req.question
        yahoo_results = search_yahoo(
            query=search_query,
            category=req.category,
            timelimit=req.timelimit,
            max_results=req.max_results,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"DDGS Yahoo search failed: {e}")

    if not yahoo_results:
        raise HTTPException(status_code=404, detail="No results returned from Yahoo backend.")

    # Step B: synthesize into A2UI v1.0 JSON via Nemotron
    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": A2UI_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"User Question: {req.question}\n\n"
                        f"Yahoo Search Results (JSON):\n{json.dumps(yahoo_results, indent=2, default=str)}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2048,
        )
        raw_content = completion.choices[0].message.content
        if not raw_content:
            raise ValueError("Model returned empty content")
        a2ui_messages = json.loads(raw_content)

        return {
            "model": MODEL_ID,
            "a2ui": a2ui_messages,
            "raw_sources_count": len(yahoo_results),
        }
    except json.JSONDecodeError as e:
        log.exception("Model returned invalid JSON")
        raise HTTPException(status_code=502, detail=f"Model returned invalid JSON: {e}")
    except Exception as e:
        log.exception("Nemotron API call failed")
        raise HTTPException(status_code=500, detail=f"Nemotron inference error: {e}")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "nvidia_key_set": bool(NVIDIA_API_KEY)}


# --- Serve the built React frontend (frontend/dist) from this same process ---
_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(_DIST, "assets")), name="assets")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(_DIST, "index.html"))
    log.info("Serving built frontend from %s", _DIST)
else:
    log.info("No frontend/dist build found — run `npm run build` in frontend/ to serve UI from this server")
