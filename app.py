import json
import logging
import os

import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

st.set_page_config(page_title="Yahoo Finance + A2UI", page_icon="📈", layout="wide")

st.title("📈 Yahoo Finance + A2UI")
st.caption("Enter a ticker — real price, financials and news from Yahoo Finance (yfinance), "
           "synthesized into a Google A2UI surface with NVIDIA Nemotron.")

# --- NVIDIA NIM (Nemotron) for A2UI synthesis ---
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
- The Column holds: a header Row (Icon + company/ticker Text), a metrics Row (2-4 bold Text
  lines with key numbers: price, change, market cap, PE...), a summary Text (2-4 sentences),
  a Divider, and a sources Text (variant "caption") listing news sources.

Grounding rules:
1. Ground every number and claim strictly in the provided data. Never invent data.
2. Every component id referenced in `child`/`children` MUST exist as a component entry.
3. Return ONLY the raw JSON array. No markdown fences, no explanation.
"""


def fetch_ticker_data(ticker: str) -> dict:
    """Pull real data for a ticker from Yahoo Finance via yfinance."""
    t = yf.Ticker(ticker)

    fast = t.fast_info
    price = fast.last_price
    if price is None:
        raise ValueError(f"No price data found for ticker '{ticker}'. Is it valid?")

    info = {}
    try:
        info = t.info or {}
    except Exception as e:
        log.warning("t.info failed for %s: %s", ticker, e)

    change_pct = None
    prev = fast.previous_close
    if prev:
        change_pct = (price - prev) / prev * 100

    news = []
    try:
        for n in (t.news or [])[:5]:
            content = n.get("content", n)
            news.append({
                "title": content.get("title") or n.get("title", ""),
                "source": (content.get("provider") or {}).get("displayName", "")
                          if isinstance(content, dict) else "",
                "url": (content.get("clickThroughUrl") or content.get("url") or {}).get("url", "")
                       if isinstance(content, dict) else "",
            })
    except Exception as e:
        log.warning("news fetch failed for %s: %s", ticker, e)

    return {
        "ticker": ticker.upper(),
        "company": info.get("longName") or info.get("shortName") or ticker.upper(),
        "currency": fast.currency or info.get("currency") or "USD",
        "price": price,
        "previous_close": prev,
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "sector": info.get("sector"),
        "business_summary": info.get("longBusinessSummary"),
        "news": news,
    }


def synthesize_a2ui(data: dict) -> list[dict]:
    """Call Nemotron via NVIDIA NIM to convert ticker data into A2UI v1.0 messages."""
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
                    "Build an A2UI finance card for this ticker data:\n"
                    f"{json.dumps(data, indent=2, default=str)}"
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
    try:
        return os.path.getmtime(_EMBED_TEMPLATE_PATH)
    except OSError:
        return 0.0


@st.cache_data(ttl=3600, show_spinner=False)
def load_embed_template(_mtime: float) -> str | None:
    """Load the built A2UI embed page; the _mtime arg busts the cache on rebuild."""
    try:
        with open(_EMBED_TEMPLATE_PATH, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def render_a2ui_iframe(a2ui_messages: list[dict], height: int = 560) -> bool:
    """Render A2UI messages via the real @a2ui/react embed page in an iframe."""
    template = load_embed_template(_embed_mtime())
    if template is None:
        return False
    payload = json.dumps(a2ui_messages).replace("</script>", "<\\/script>")
    html = template.replace("__A2UI_MESSAGES__", payload)
    components.html(html, height=height, scrolling=True)
    return True


# --- Input ---
ticker = st.text_input(
    "Company ticker",
    value="MELI",
    placeholder="e.g. MELI, AAPL, TSLA",
).strip().upper()

if st.button("Fetch & render A2UI", type="primary") or ticker:
    if not ticker:
        st.info("Enter a ticker above.")
    else:
        try:
            with st.spinner(f"Fetching {ticker} from Yahoo Finance..."):
                data = fetch_ticker_data(ticker)
        except ValueError as e:
            data = None
            st.error(f"❓ {e}")
        except Exception as e:
            data = None
            st.error(f"💥 Could not fetch '{ticker}' (`{type(e).__name__}`): `{e}`")

        if data:
            chg = data.get("change_pct")
            chg_str = f" ({chg:+.2f}%)" if chg is not None else ""
            st.success(
                f"**{data['company']}** ({data['ticker']}) — "
                f"{data['price']:.2f} {data['currency']}{chg_str}"
            )

            if st.button("✨ Render as A2UI", type="secondary", use_container_width=True):
                try:
                    with st.spinner(f"Synthesizing A2UI with {MODEL_ID}..."):
                        a2ui_messages = synthesize_a2ui(data)
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

            # --- Raw data view ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Price", f"{data['price']:.2f} {data['currency']}",
                      f"{chg:+.2f}%" if chg is not None else None)
            m2.metric("Market Cap", f"{data['market_cap']:,}" if data.get("market_cap") else "—")
            m3.metric("P/E", f"{data['pe_ratio']:.1f}" if data.get("pe_ratio") else "—")
            m4.metric("Sector", data.get("sector") or "—")

            if data.get("news"):
                with st.expander(f"📰 Latest news ({len(data['news'])})"):
                    for n in data["news"]:
                        if n.get("url"):
                            st.markdown(f"- [{n['title']}]({n['url']})" + (f" — {n['source']}" if n["source"] else ""))
                        else:
                            st.markdown(f"- {n['title']}")

            if data.get("business_summary"):
                with st.expander("🏢 Business summary"):
                    st.write(data["business_summary"])
