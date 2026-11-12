import logging
import time

import requests
import streamlit as st
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

    backend = st.selectbox(
        "Backend (used strictly)",
        available,
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
