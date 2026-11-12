import json
import logging
import os
import time

import requests
import streamlit as st
import streamlit.components.v1 as components
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

# --- Make ddgs engine errors visible in the terminal ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

st.set_page_config(page_title="DDGS Live Search Tester", page_icon="🔎", layout="wide")

st.title("🔎 DDGS Free Web Search Tester")
st.caption("The selected backend is used strictly — no random substitution, no hidden fallback. "
           "Results are tagged with the backend that served them.")

# Backends discovered from the INSTALLED ddgs engine registry — only engines that
# exist AND are enabled appear here, so selecting one can never silently fall back to auto.
from ddgs.engines import ENGINES as _DDGS_ENGINES

TEXT_BACKENDS = sorted(_DDGS_ENGINES.get("text", {}).keys())
NEWS_BACKENDS = sorted(_DDGS_ENGINES.get("news", {}).keys())


def get_outbound_ip() -> str:
    """Check which public IP this machine's requests are leaving from."""
    try:
        return requests.get("https://api.ipify.org", timeout=5).text.strip()
    except Exception:
        return "Unknown"


# --- NVIDIA NIM (Nemotron) client for A2UI synthesis ---
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_ID = os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")

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

Component rules (A2UI v1.0 Basic Catalog - these are the ONLY components):
- Card: {"id": "...", "component": "Card", "child": "<single child component id>"}
  Card takes ONE child. To put multiple things in a card, wrap them in a Column/Row and
  pass that container's id as `child`.
- Column / Row / List: {"id": "...", "component": "Column", "children": ["<id>", "<id>", ...]}
  `children` is an ARRAY OF COMPONENT IDS. Components CANNOT be nested inline - every child
  must be its own entry in the components list, referenced by id.
- Text: {"id": "...", "component": "Text", "text": "<literal string>", "variant": "body"|"caption"}
  Use variant "caption" for smaller source lines. Simple markdown (bold) is supported.
- Icon: {"id": "...", "component": "Icon", "name": "<material icon name>"}
- Divider: {"id": "...", "component": "Divider"}

Layout guidance for a finance answer:
- root: a Card whose child is a Column.
- The Column holds: a header Row (Icon + title Text), a metrics Row (2-4 bold Text lines
  with key numbers), a summary Text (2-4 sentences), a Divider, and a sources Text
  (variant "caption") listing the source URLs from the search results.

Grounding rules:
1. Ground every number and claim strictly in the provided search results. Never invent data.
2. Every component id referenced in `child`/`children` MUST exist as a component entry.
3. Return ONLY the raw JSON array. No markdown fences, no explanation.
"""


def synthesize_a2ui(question: str, results: list[dict]) -> list[dict]:
    """Call Nemotron via NVIDIA NIM to convert search results into A2UI v1.0 messages."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("The 'openai' package is not installed. Run: pip install openai")

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set. Set it before asking for an A2UI answer.")

    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": A2UI_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User Question: {question}\n\n"
                    f"Yahoo Search Results (JSON):\n{json.dumps(results, indent=2, default=str)}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=4096,
        # Nemotron reasoning mode: thinking output would otherwise fill the
        # response (and can leave `content` empty). Disable it explicitly.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = completion.choices[0]
    msg = choice.message
    raw = msg.content
    log.info(
        "Nemotron call: finish_reason=%s content_len=%s reasoning_present=%s usage=%s",
        choice.finish_reason,
        len(raw) if raw else 0,
        bool(getattr(msg, "reasoning_content", None) or getattr(msg, "model_extra", {}).get("reasoning_content")),
        getattr(completion, "usage", None),
    )
    if not raw:
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "model_extra", {}).get(
            "reasoning_content"
        )
        if reasoning:
            # The model answered inside its thinking channel — try to pull the JSON out.
            start, end = reasoning.find("["), reasoning.rfind("]")
            if start != -1 and end > start:
                raw = reasoning[start : end + 1]
        if not raw:
            detail = f"finish_reason={choice.finish_reason}"
            if reasoning:
                detail += f", reasoning_content started with: {reasoning[:120]!r}"
            raise RuntimeError(
                f"Model returned empty content ({detail}). "
                f"If finish_reason=length, raise max_tokens in app.py."
            )
    parsed = json.loads(raw)
    # The model must return an array of A2UI messages; tolerate a single wrapped object.
    if isinstance(parsed, dict):
        for key in ("messages", "a2ui", "surfaceUpdate"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        return [parsed]
    return parsed


# --- A2UI iframe renderer (built once from frontend/, real @a2ui/react components) ---
_EMBED_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "frontend", "dist", "embed.html"
)


def _embed_mtime() -> float:
    """Mtime of the embed build, so the cache invalidates when it's rebuilt."""
    try:
        return os.path.getmtime(_EMBED_TEMPLATE_PATH)
    except OSError:
        return 0.0


@st.cache_data(ttl=3600, show_spinner=False)
def load_embed_template(_mtime: float) -> str | None:
    """Load the built A2UI embed page (with a __A2UI_MESSAGES__ placeholder).

    The _mtime argument (not used directly) busts the cache whenever the
    embed build appears or is rebuilt.
    """
    try:
        with open(_EMBED_TEMPLATE_PATH, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def render_a2ui_iframe(a2ui_messages: list[dict], height: int = 560) -> bool:
    """Render A2UI messages via the real @a2ui/react embed page in an iframe.

    Returns False if the embed build is missing.
    """
    template = load_embed_template(_embed_mtime())
    if template is None:
        return False
    payload = json.dumps(a2ui_messages).replace("</script>", "<\\/script>")
    html = template.replace("__A2UI_MESSAGES__", payload)
    components.html(html, height=height, scrolling=True)
    return True


def search_one(backend: str, query: str, category: str, region: str,
               timelimit: str | None, max_results: int, proxy: str | None,
               retries: int = 3) -> list[dict]:
    """Query exactly ONE backend — the selected one — and tag each result with it.

    Retries on transient DNS errors (flaky VPN resolvers intermittently refuse
    queries; an immediate retry usually succeeds).
    """
    ddgs = DDGS(proxy=proxy or None, timeout=15)
    kwargs = dict(query=query, region=region, timelimit=timelimit, max_results=max_results, backend=backend)
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if category == "News":
                results = ddgs.news(**kwargs)
            else:
                results = ddgs.text(**kwargs)
            for r in results:
                r["_backend"] = backend  # traceability: which engine served this result
            if attempt > 1:
                log.info("backend=%s query=%r -> %d results (succeeded on retry %d)",
                         backend, query, len(results), attempt)
            log.info("backend=%s category=%s query=%r -> %d results", backend, category, query, len(results))
            return results
        except (DDGSException, TimeoutException) as e:
            last_exc = e
            # Only retry transient DNS failures — a ratelimit won't fix itself in a second
            is_dns = "DNSError" in str(e) or "Query Refused" in str(e)
            if attempt < retries and is_dns:
                log.warning("backend=%s attempt %d/%d failed (transient DNS): %.80s — retrying...",
                            backend, attempt, retries, str(e))
                time.sleep(1.0 * attempt)
            elif attempt < retries:
                raise
    raise last_exc  # type: ignore[misc]


# --- Sidebar controls ---
with st.sidebar:
    st.subheader("⚙️ Search controls")
    category = st.radio("Category", ["Web (text)", "News"], horizontal=True)
    available = NEWS_BACKENDS if category == "News" else TEXT_BACKENDS

    default_backend = "yahoo" if "yahoo" in available else (available[0] if available else "duckduckgo")
    backend = st.selectbox(
        "Backend (used strictly)",
        available,
        index=available.index(default_backend),
        help="Only this backend is queried. If it fails, switch to another one here — nothing is selected for you.",
    )

    timelimit = st.selectbox(
        "Time Limit",
        options=[None, "d", "w", "m", "y"],
        index=3,
        format_func=lambda x: {
            None: "Any time", "d": "Past day (d)", "w": "Past week (w)",
            "m": "Past month (m)", "y": "Past year (y)",
        }[x],
    )
    max_results = st.slider("Max Results", min_value=1, max_value=25, value=10)
    region = st.text_input("Region", value="us-en")
    proxy = st.text_input("Optional Proxy URL", placeholder="http://user:pass@ip:port or socks5://...")

    st.divider()
    st.markdown(f"**Outbound IP:** `{get_outbound_ip()}`")

    st.divider()
    st.subheader("🩺 Backend health check")
    if st.button("Test all backends", use_container_width=True):
        st.caption("Querying every backend with a 1-result probe...")
        rows = []
        for b in available:
            t0 = time.monotonic()
            try:
                r = search_one(b, "hello world", category, region, None, 1, proxy.strip() or None, retries=1)
                dt = time.monotonic() - t0
                rows.append((b, f"✅ {len(r)} result(s) in {dt:.1f}s" if r else f"⚠️ 0 results in {dt:.1f}s"))
            except Exception as e:
                rows.append((b, f"❌ {type(e).__name__}: {str(e)[:60]}"))
        st.markdown("\n".join(f"- **{b}** — {s}" for b, s in rows))

# --- Main search bar ---
query = st.text_input(
    "Enter your search query",
    value="MercadoLibre revenue growth",
    placeholder="Type anything and press Enter...",
)

if st.button("Search", type="primary") or query:
    start_t = time.monotonic()
    try:
        with st.spinner(f"Querying DDGS (backend='{backend}')..."):
            results = search_one(backend, query, category, region, timelimit, max_results, proxy.strip() or None)
    except RatelimitException as e:
        results = None
        st.error(f"🚫 **RatelimitException** from `{backend}` after {time.monotonic() - start_t:.2f}s: `{e}`")
        st.info(f"`{backend}` blocked this request. Pick a different **Backend** in the sidebar and search again.")
    except TimeoutException as e:
        results = None
        st.warning(f"⏱️ **TimeoutException** from `{backend}` after {time.monotonic() - start_t:.2f}s: `{e}`")
    except DDGSException as e:
        results = None
        st.error(f"⚠️ **DDGSException** from `{backend}` after {time.monotonic() - start_t:.2f}s: `{e}`")
        st.info(f"`{backend}` returned nothing. Pick a different **Backend** in the sidebar and search again.")
    except Exception as e:
        results = None
        st.error(f"💥 **Unexpected Error** from `{backend}` (`{type(e).__name__}`): `{e}`")
    elapsed = time.monotonic() - start_t

    if results is not None:
        if not results:
            st.warning(f"0 results returned from `{backend}` in {elapsed:.2f}s (no exception raised).")
        else:
            st.success(f"Fetched {len(results)} results in {elapsed:.2f}s — served by **{backend}**")

            # --- AI answer rendered as a real A2UI surface (Nemotron + @a2ui/react) ---
            if st.button("✨ Ask AI — render as A2UI", type="secondary", use_container_width=True):
                try:
                    with st.spinner(f"Synthesizing A2UI with {MODEL_ID}..."):
                        a2ui_messages = synthesize_a2ui(query, results)
                    if not render_a2ui_iframe(a2ui_messages):
                        st.error(
                            "A2UI embed build not found. Build it once with: "
                            "`cd frontend && npm install && npm run build`"
                        )
                    with st.expander("📦 Raw A2UI JSON"):
                        st.json(a2ui_messages)
                except RuntimeError as e:
                    st.error(f"🔑 {e}")
                except json.JSONDecodeError as e:
                    st.error(f"⚠️ Model returned invalid JSON: {e}")
                except Exception as e:
                    st.error(f"💥 AI synthesis failed (`{type(e).__name__}`): `{e}`")

            for i, r in enumerate(results, 1):
                url = r.get("href") or r.get("url") or "#"
                title = r.get("title") or "(Untitled)"
                body = r.get("body") or ""
                date_str = r.get("date") or ""
                source_str = r.get("source") or ""

                with st.container(border=True):
                    st.markdown(f"**{i}. [{title}]({url})**")
                    meta = [f"🏷️ backend: `{r.get('_backend', backend)}`"]
                    if date_str:
                        meta.append(f"📅 {date_str}")
                    if source_str:
                        meta.append(f"📰 {source_str}")
                    st.caption(" | ".join(meta))
                    if body:
                        st.write(body)

            with st.expander("📦 Raw JSON Response"):
                st.json(results)
