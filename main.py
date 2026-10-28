import os
import json
import re
import asyncio
import sqlite3
from datetime import datetime, timezone
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="MuniRadar Credit Intel API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BRAVE_API_KEY     = os.environ.get("BRAVE_API_KEY", "")
DEFAULT_MODEL     = os.environ.get("DEFAULT_MODEL", "claude-sonnet-4-6")
DB_PATH           = os.environ.get("MUNIRADAR_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "muniradar.db"))

SYSTEM_PROMPT = """You are a senior municipal credit analyst at The Tribeca Group, a professional fixed income portfolio management team at Morgan Stanley. Your job is to produce rigorous, opinionated credit intelligence cards for muni bond CUSIPs. You have deep expertise in municipal credit structures, revenue pledge analysis, intercept mechanisms, sector benchmarks, and investability frameworks.

CRITICAL RULE — NEVER GUESS THE ISSUER OR RATINGS: All issuer identity and ratings data is provided to you from Bloomberg in the ## Bloomberg Data section. Use ONLY that data for issuer name, ratings, maturity, coupon, and pricing. Do NOT override, second-guess, or supplement Bloomberg data with your training data for these fields. For recent developments and news, use the live web data provided. For credit analysis, sector context, and investability judgment, use your expertise.

Return ONLY valid JSON matching this exact schema — no preamble, no markdown, no explanation:
{
  "issuer": {
    "name": "Full official issuer name",
    "sector": "GO | Water & Sewer | Electric | School District | Transportation | Healthcare | Higher Ed | Housing | Special Tax | Airport | Other",
    "bond_type": "General Obligation | Revenue | COP/Lease | Special Assessment | Other",
    "state": "Two-letter state code",
    "service_area": "Geographic area served / jurisdiction description",
    "description": "2-3 sentence description of what this entity is, what it does, and its credit profile at a high level"
  },
  "revenue_sources": [
    { "source": "Revenue source name", "share": "~XX% (or qualitative)", "notes": "Brief note on stability/pledging" }
  ],
  "security_pledge": "Describe the legal pledge securing these bonds — property tax levy, net revenues, rate covenant, state intercept, etc. Be specific about priority.",
  "key_metrics": [
    { "metric": "Metric name", "value": "Reported value with units", "vs_median": "above | below | in-line", "notes": "Context vs sector median" }
  ],
  "strengths": ["Strength 1 — specific, factual", "Strength 2", "Strength 3"],
  "risks": ["Risk 1 — specific, factual", "Risk 2", "Risk 3"],
  "recent_developments": [
    { "date": "Mon YYYY", "headline": "Short factual headline", "impact": "positive | negative | neutral", "detail": "1-2 sentences with specifics" }
  ],
  "ratings": {
    "moodys": { "rating": "Aa1 | etc | NR", "outlook": "Stable | Positive | Negative | Watch | NR", "last_action": "Brief description of last rating action" },
    "sp": { "rating": "AAA | etc | NR", "outlook": "Stable | Positive | Negative | CreditWatch | NR", "last_action": "Brief description" },
    "fitch": { "rating": "AAA | etc | NR", "outlook": "Stable | Positive | Negative | Watch | NR", "last_action": "Brief description" }
  },
  "investability": {
    "1yr":  { "verdict": "INVESTABLE | CAUTIOUS | AVOID", "rationale": "One sentence rationale specific to this tenor" },
    "5yr":  { "verdict": "INVESTABLE | CAUTIOUS | AVOID", "rationale": "One sentence rationale" },
    "10yr": { "verdict": "INVESTABLE | CAUTIOUS | AVOID", "rationale": "One sentence rationale" },
    "15yr": { "verdict": "INVESTABLE | CAUTIOUS | AVOID", "rationale": "One sentence rationale" },
    "20yr": { "verdict": "INVESTABLE | CAUTIOUS | AVOID", "rationale": "One sentence rationale" }
  },
  "overall_recommendation": "BUY | HOLD | REDUCE | AVOID",
  "analyst_note": "2-3 sentence senior analyst take: the single most important thing to know about this credit right now, and what would change the view. Be direct and opinionated.",
  "data_note": "Brief disclosure about data currency and sources used in this analysis — include whether Bloomberg and live web data was retrieved"
}

Be specific to the actual issuer. Use real figures where known. Do not give generic muni boilerplate. If data is uncertain, say so in the notes fields rather than omitting or fabricating. Distinguish between bond types within an issuer (e.g. LADWP Power vs Water; LAUSD GO vs COPs). When live web data is provided in the prompt, prioritize it over your training data for recent developments."""


# ── Bloomberg data fetch (synchronous, runs in thread pool) ──────────────────

BLP_FIELDS = [
    "ISSUER",
    "SECURITY_DES",
    "RTG_MOODY",
    "RTG_SP",
    "RTG_FITCH",
    "NXT_CALL_DT",
    "MATURITY",
    "CPN",
    "YLD_YTM_MID",
    "PX_MID",
    "TAX_STATUS",
    "STATE_CODE",
    "COLLAT_TYP",
    "ISSUE_DT",
    "AMT_OUTSTANDING",
]

def _fetch_bloomberg_sync(cusip: str) -> dict:
    """Synchronous Bloomberg fetch — called in a thread pool."""
    try:
        import blpapi  # type: ignore
        session_options = blpapi.SessionOptions()
        session_options.setServerHost("localhost")
        session_options.setServerPort(8194)

        session = blpapi.Session(session_options)
        if not session.start():
            raise RuntimeError("Bloomberg session failed to start")
        if not session.openService("//blp/refdata"):
            session.stop()
            raise RuntimeError("Bloomberg refdata service unavailable")

        service  = session.getService("//blp/refdata")
        request  = service.createRequest("ReferenceDataRequest")
        security = f"/cusip/{cusip}"
        request.append("securities", security)
        for f in BLP_FIELDS:
            request.append("fields", f)

        session.sendRequest(request)

        result = {}
        while True:
            ev = session.nextEvent(timeout=5000)
            for msg in ev:
                if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                    sd = msg.getElement("securityData")
                    for i in range(sd.numValues()):
                        sec_el = sd.getValueAsElement(i)
                        fd = sec_el.getElement("fieldData")
                        for field in BLP_FIELDS:
                            try:
                                val = fd.getElementAsString(field)
                                if val and val not in ("N.A.", "N/A", ""):
                                    result[field] = val
                            except Exception:
                                pass
            if ev.eventType() == blpapi.Event.RESPONSE:
                break

        session.stop()
        return result

    except ImportError:
        raise RuntimeError("blpapi not installed on this machine")
    except Exception as e:
        raise RuntimeError(f"Bloomberg error: {e}")


async def fetch_bloomberg(cusip: str) -> dict:
    """Run Bloomberg fetch in a thread pool (non-blocking)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_bloomberg_sync, cusip)


def format_bloomberg_data(data: dict) -> str:
    """Format Bloomberg fields into a prompt section."""
    if not data:
        return ""
    lines = ["## Bloomberg Data (verified — use as ground truth for issuer and ratings)"]
    label_map = {
        "ISSUER":           "Issuer name",
        "SECURITY_DES":     "Security description",
        "RTG_MOODY":        "Moody's rating",
        "RTG_SP":           "S&P rating",
        "RTG_FITCH":        "Fitch rating",
        "NXT_CALL_DT":      "Next call date",
        "MATURITY":         "Maturity date",
        "CPN":              "Coupon",
        "YLD_YTM_MID":      "Yield to maturity (mid)",
        "PX_MID":           "Price (mid)",
        "TAX_STATUS":       "Tax status",
        "STATE_CODE":       "State",
        "COLLAT_TYP":       "Collateral / security type",
        "ISSUE_DT":         "Issue date",
        "AMT_OUTSTANDING":  "Amount outstanding",
    }
    for field, label in label_map.items():
        if field in data:
            lines.append(f"  {label}: {data[field]}")
    return "\n".join(lines)


# ── Web enrichment helpers ────────────────────────────────────────────────────

async def brave_search(client: httpx.AsyncClient, query: str, count: int = 5) -> list[dict]:
    if not BRAVE_API_KEY:
        return []
    try:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
            params={"q": query, "count": count, "text_decorations": False},
            timeout=10.0,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = []
        for r in data.get("web", {}).get("results", []):
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "snippet": r.get("description", ""),
            })
        return results
    except Exception:
        return []


async def emma_trade_data(client: httpx.AsyncClient, cusip: str) -> str:
    try:
        url = f"https://emma.msrb.org/api/TradeActivity/GetTradeActivityByCusip?cusip={cusip}&startDate=&endDate=&sortField=tradeDate&sortOrder=desc&page=1&pageSize=10"
        resp = await client.get(url, timeout=10.0, follow_redirects=True)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        trades = data.get("TradeActivityResult", {}).get("TradeActivityItems", [])
        if not trades:
            return ""
        lines = ["Recent MSRB EMMA trade data:"]
        for t in trades[:8]:
            date  = t.get("TradeDate", "")[:10]
            price = t.get("Price", "N/A")
            yld   = t.get("Yield", "N/A")
            par   = t.get("ParTraded", "N/A")
            side  = t.get("BuySellIndicator", "")
            lines.append(
                f"  {date} | {side} | Par: ${par:,} | Price: {price} | Yield: {yld}%"
                if isinstance(par, (int, float))
                else f"  {date} | {side} | Par: {par} | Price: {price} | Yield: {yld}%"
            )
        return "\n".join(lines)
    except Exception:
        return ""


def format_search_results(results: list[dict], label: str) -> str:
    if not results:
        return ""
    lines = [f"{label}:"]
    for r in results:
        lines.append(f"  • {r['title']}")
        if r["snippet"]:
            lines.append(f"    {r['snippet'][:200]}")
        lines.append(f"    Source: {r['url']}")
    return "\n".join(lines)


async def gather_web_data(cusip: str, issuer_name: str) -> str:
    """Run web enrichment in parallel, hard cap 14s. Uses confirmed issuer name for search."""
    async def _fetch_all():
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=4.0)) as client:
            search_name   = issuer_name if issuer_name else cusip
            ratings_query = f'"{search_name}" rating action Moody\'s S&P Fitch muni 2024 2025 2026'
            news_query    = f'"{search_name}" municipal bond news 2025 2026'
            credit_query  = f'"{search_name}" municipal credit financial audit 2024 2025'
            return await asyncio.gather(
                brave_search(client, ratings_query, count=3),
                brave_search(client, news_query,    count=3),
                brave_search(client, credit_query,  count=3),
                emma_trade_data(client, cusip),
                return_exceptions=True,
            )

    try:
        results = await asyncio.wait_for(_fetch_all(), timeout=14.0)
    except asyncio.TimeoutError:
        return ""

    ratings_results, news_results, credit_results, emma_trades = [
        r if not isinstance(r, Exception) else ([] if i < 3 else "")
        for i, r in enumerate(results)
    ]

    sections = []
    if emma_trades:
        sections.append(emma_trades)

    for block in [
        format_search_results(ratings_results, "Recent web: rating actions"),
        format_search_results(news_results,    "Recent web: news & developments"),
        format_search_results(credit_results,  "Recent web: credit & financial"),
    ]:
        if block:
            sections.append(block)

    if not sections:
        return ""

    return "\n\n## Live Web Data Retrieved at Request Time\n" + "\n\n".join(sections) + "\n"


# ── Saved cards library (SQLite) ──────────────────────────────────────────────

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# One card per ISSUER. Every CUSIP ever searched is linked to its issuer card.
# A CUSIP finds its card by: (1) exact CUSIP already linked, (2) same 6-char
# issuer prefix as a linked CUSIP, (3) same Bloomberg ISSUER name.

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (name or "").upper())


def init_db():
    with _db() as conn:
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(cards)").fetchall()]
        old_rows = []
        if "cusip" in cols:  # v1 schema (one row per CUSIP) -> migrate
            old_rows = conn.execute("SELECT cusip, intel_json, saved_at FROM cards ORDER BY id").fetchall()
            conn.execute("DROP TABLE cards")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cards (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                issuer         TEXT,
                bbg_issuer     TEXT,
                bbg_norm       TEXT,
                state          TEXT,
                sector         TEXT,
                recommendation TEXT,
                intel_json     TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS card_cusips (
                cusip    TEXT PRIMARY KEY,
                prefix   TEXT NOT NULL,
                card_id  INTEGER NOT NULL,
                added_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cc_prefix ON card_cusips(prefix)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cards_norm ON cards(bbg_norm)")
        for r in old_rows:
            cid = find_card(conn, r["cusip"])
            upsert_card(conn, cid, r["cusip"], "", json.loads(r["intel_json"]))


def find_card(conn, cusip: str, bbg_issuer: str = "") -> Optional[int]:
    row = conn.execute("SELECT card_id FROM card_cusips WHERE cusip = ?", (cusip,)).fetchone()
    if not row:
        row = conn.execute("SELECT card_id FROM card_cusips WHERE prefix = ? LIMIT 1", (cusip[:6],)).fetchone()
    if not row and _norm(bbg_issuer):
        row = conn.execute("SELECT id AS card_id FROM cards WHERE bbg_norm = ? LIMIT 1", (_norm(bbg_issuer),)).fetchone()
    return row["card_id"] if row else None


def link_cusip(conn, card_id: int, cusip: str):
    conn.execute(
        "INSERT OR REPLACE INTO card_cusips (cusip, prefix, card_id, added_at) VALUES (?,?,?, COALESCE((SELECT added_at FROM card_cusips WHERE cusip = ?), ?))",
        (cusip, cusip[:6], card_id, cusip, _now()),
    )


def upsert_card(conn, card_id: Optional[int], cusip: str, bbg_issuer: str, intel: dict) -> int:
    issuer = intel.get("issuer") or {}
    vals = (issuer.get("name"), issuer.get("state"), issuer.get("sector"),
            intel.get("overall_recommendation"), json.dumps(intel), _now())
    if card_id:
        conn.execute(
            "UPDATE cards SET issuer=?, state=?, sector=?, recommendation=?, intel_json=?, updated_at=? WHERE id=?",
            vals + (card_id,),
        )
        if bbg_issuer:
            conn.execute("UPDATE cards SET bbg_issuer=?, bbg_norm=? WHERE id=?", (bbg_issuer, _norm(bbg_issuer), card_id))
    else:
        cur = conn.execute(
            "INSERT INTO cards (issuer, state, sector, recommendation, intel_json, updated_at, created_at, bbg_issuer, bbg_norm) VALUES (?,?,?,?,?,?,?,?,?)",
            vals + (vals[-1], bbg_issuer, _norm(bbg_issuer)),
        )
        card_id = cur.lastrowid
    link_cusip(conn, card_id, cusip)
    return card_id


def card_payload(conn, card_id: int, cached: bool) -> Optional[dict]:
    row = conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
    if not row:
        return None
    cusips = [r["cusip"] for r in conn.execute(
        "SELECT cusip FROM card_cusips WHERE card_id = ? ORDER BY added_at, cusip", (card_id,)).fetchall()]
    intel = json.loads(row["intel_json"])
    intel["library_meta"] = {
        "id": row["id"],
        "saved_at": row["updated_at"],
        "cusips": cusips,
        "cached": cached,
    }
    return intel


init_db()


# ── Request model ─────────────────────────────────────────────────────────────

class CreditIntelRequest(BaseModel):
    cusip: str
    pdf_text: Optional[str] = None
    model: Optional[str] = None
    refresh: Optional[bool] = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "MuniRadar Credit Intel API",
        "web_enrichment": bool(BRAVE_API_KEY),
        "bloomberg": "required",
    }


@app.post("/api/credit-intel")
async def credit_intel(req: CreditIntelRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    cusip = req.cusip.strip().upper()
    model = req.model or DEFAULT_MODEL

    # ── Issuer card already in library? Return it (no Bloomberg / API call) ──
    if not req.refresh:
        with _db() as conn:
            cid = find_card(conn, cusip)
            if cid:
                link_cusip(conn, cid, cusip)
                cached = card_payload(conn, cid, cached=True)
                if cached:
                    return cached

    # ── Bloomberg fetch — required, no fallback ──
    try:
        blp_data = await asyncio.wait_for(fetch_bloomberg(cusip), timeout=10.0)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Bloomberg timed out. Ensure the Bloomberg Terminal is running and blpapi is installed."
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not blp_data:
        raise HTTPException(
            status_code=404,
            detail=f"Bloomberg returned no data for CUSIP {cusip}. Verify the CUSIP is correct."
        )

    bloomberg_section = format_bloomberg_data(blp_data)
    issuer_name = blp_data.get("ISSUER", "")

    # ── Same issuer under a different CUSIP prefix? (matched on Bloomberg issuer name) ──
    with _db() as conn:
        existing_card_id = find_card(conn, cusip, issuer_name)
        if existing_card_id and not req.refresh:
            link_cusip(conn, existing_card_id, cusip)
            cached = card_payload(conn, existing_card_id, cached=True)
            if cached:
                return cached

    # ── Web enrichment (uses confirmed issuer name for better search) ──
    web_data = await gather_web_data(cusip, issuer_name)

    pdf_section = ""
    if req.pdf_text:
        pdf_section = f"\n\n## Uploaded Research / Document\n{req.pdf_text[:12000]}\n"

    user_message = f"""Analyze CUSIP: {cusip}

{bloomberg_section}
{web_data}{pdf_section}

Generate a complete credit intelligence card using the Bloomberg data above as the authoritative source for issuer identity and ratings. Use the live web data for recent developments. Apply your municipal credit expertise for sector context, financial analysis, and investability judgments.

Key areas to address:
1. Use ONLY the Bloomberg issuer name — do not alter or guess it
2. Revenue sources and security pledge — infer from security type and sector
3. Key financial metrics vs sector benchmarks (DSCR, debt/revenue, days cash, etc. as applicable)
4. Credit strengths and risks — be candid and specific
5. Recent developments — prioritize live web data for anything 2024 or later
6. Ratings — use EXACTLY what Bloomberg returned; note outlook from web data if available
7. Investability by tenor — be granular
8. In data_note, confirm Bloomberg data was used as primary source"""

    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 4000,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_message}],
            },
        )

    if response.status_code != 200:
        err = response.json().get("error", {})
        raise HTTPException(status_code=response.status_code, detail=err.get("message", "Anthropic API error"))

    data = response.json()
    raw  = data["content"][0]["text"].strip()

    match = re.search(r'\{[\s\S]*\}', raw)
    if not match:
        raise HTTPException(status_code=500, detail="Could not parse model response as JSON")

    try:
        intel = json.loads(match.group())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error: {e}")

    # Save to library — one card per issuer (refresh overwrites the existing card)
    try:
        with _db() as conn:
            cid = upsert_card(conn, existing_card_id, cusip, issuer_name, intel)
            saved = card_payload(conn, cid, cached=False)
            if saved:
                return saved
    except Exception:
        pass

    return intel


@app.get("/api/cards")
def list_cards():
    with _db() as conn:
        rows = conn.execute("""
            SELECT c.id, c.issuer, c.state, c.sector, c.recommendation, c.updated_at AS saved_at,
                   (SELECT GROUP_CONCAT(cusip, ',') FROM
                       (SELECT cusip FROM card_cusips WHERE card_id = c.id ORDER BY added_at, cusip)) AS cusips
            FROM cards c
            ORDER BY c.updated_at DESC, c.id DESC
        """).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["cusips"] = d["cusips"].split(",") if d["cusips"] else []
        out.append(d)
    return out


@app.get("/api/cards/{card_id}")
def get_card(card_id: int):
    with _db() as conn:
        intel = card_payload(conn, card_id, cached=True)
    if not intel:
        raise HTTPException(status_code=404, detail="Card not found")
    cusips = intel["library_meta"]["cusips"]
    return {"cusip": cusips[0] if cusips else "", "intel": intel}


@app.delete("/api/cards/{card_id}")
def delete_card(card_id: int):
    with _db() as conn:
        cur = conn.execute("DELETE FROM cards WHERE id = ?", (card_id,))
        conn.execute("DELETE FROM card_cusips WHERE card_id = ?", (card_id,))
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Card not found")
    return {"deleted": card_id}
