"""
MuniRadar — local backend (runs on the Mac with Bloomberg)
==========================================================
  GET    /  or /api/health    status (the page itself is on Netlify)
  POST   /api/lookup          Bloomberg issuer lookup + "is there a saved report?"
  POST   /api/report          build a new report (streams sections as they finish)
  GET    /api/issuers         saved issuer library
  GET    /api/issuers/{id}    one saved issuer report
  DELETE /api/issuers/{id}    remove a saved issuer
"""

import os
import re
import json
import time
import asyncio
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DB_PATH           = os.environ.get("MUNIRADAR_DB", os.path.join(BASE_DIR, "muniradar.db"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Bloomberg data fetch (synchronous, runs in thread pool) ──────────────────
# Credit-relevant fields only. No price / yield / coupon / maturity: this is an
# issuer-level credit tool, and market data stays on the terminal.
# ULT_BORROWER* = the actual borrower on conduit deals (Bloomberg ignores
# whichever spelling it doesn't recognize).

BLP_FIELDS = [
    "ISSUER",
    "ULT_BORROWER",
    "ULT_BORROWER_NAME",
    "SECURITY_DES",
    "RTG_MOODY",
    "RTG_SP",
    "RTG_FITCH",
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


async def get_bloomberg(cusip: str) -> dict:
    """Bloomberg fetch with friendly errors. Required — no fallback."""
    try:
        data = await asyncio.wait_for(fetch_bloomberg(cusip), timeout=10.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503,
                            detail="Bloomberg timed out. Ensure the Bloomberg Terminal is running and signed in.")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if not data:
        raise HTTPException(status_code=404,
                            detail=f"Bloomberg returned no data for CUSIP {cusip}. Verify the CUSIP is correct.")
    return data


def borrower_name(bbg: dict) -> str:
    return bbg.get("ULT_BORROWER") or bbg.get("ULT_BORROWER_NAME") or ""


BBG_LABELS = {
    "ISSUER": "Issuer name",
    "ULT_BORROWER": "Ultimate borrower / obligor",
    "ULT_BORROWER_NAME": "Ultimate borrower / obligor",
    "SECURITY_DES": "Security description",
    "RTG_MOODY": "Moody's rating",
    "RTG_SP": "S&P rating",
    "RTG_FITCH": "Fitch rating",
    "TAX_STATUS": "Tax status",
    "STATE_CODE": "State",
    "COLLAT_TYP": "Collateral / security type",
    "ISSUE_DT": "Issue date",
    "AMT_OUTSTANDING": "Amount outstanding",
}


def format_bloomberg_data(data: dict) -> str:
    """Format Bloomberg fields into a prompt section."""
    if not data:
        return ""
    lines = ["## Bloomberg Data (verified — ground truth for issuer identity and current ratings)"]
    for field, label in BBG_LABELS.items():
        if field in data:
            lines.append(f"  {label}: {data[field]}")
    return "\n".join(lines)


# Short-lived cache so "Pull new" right after a lookup doesn't hit Bloomberg twice
_BBG_CACHE: dict = {}

async def get_bloomberg_cached(cusip: str) -> dict:
    hit = _BBG_CACHE.get(cusip)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    data = await get_bloomberg(cusip)
    _BBG_CACHE[cusip] = (time.time(), data)
    return data


# ── Issuer library (SQLite) — one saved report per issuer ────────────────────

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS issuers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                issuer_key  TEXT UNIQUE,
                name        TEXT,
                state       TEXT,
                sector      TEXT,
                report_json TEXT NOT NULL,
                bbg_json    TEXT,
                seconds     REAL,
                saved_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS issuer_cusips (
                cusip      TEXT PRIMARY KEY,
                issuer_id  INTEGER NOT NULL,
                added_at   TEXT NOT NULL
            );
        """)


init_db()


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def issuer_key(bbg: dict) -> str:
    """Borrower on conduit deals, otherwise Bloomberg ISSUER."""
    return norm(borrower_name(bbg) or bbg.get("ISSUER", ""))


def find_issuer_by_cusip(conn, cusip: str) -> Optional[int]:
    row = conn.execute("SELECT issuer_id FROM issuer_cusips WHERE cusip = ?", (cusip,)).fetchone()
    return row["issuer_id"] if row else None


def find_issuer_by_key(conn, key: str) -> Optional[int]:
    if not key:
        return None
    row = conn.execute("SELECT id FROM issuers WHERE issuer_key = ?", (key,)).fetchone()
    return row["id"] if row else None


def link_cusip(conn, issuer_id: int, cusip: str):
    conn.execute(
        "INSERT INTO issuer_cusips (cusip, issuer_id, added_at) VALUES (?,?,?) "
        "ON CONFLICT(cusip) DO UPDATE SET issuer_id = excluded.issuer_id",
        (cusip, issuer_id, now_iso()),
    )


def issuer_payload(conn, issuer_id: int) -> Optional[dict]:
    r = conn.execute("SELECT * FROM issuers WHERE id = ?", (issuer_id,)).fetchone()
    if not r:
        return None
    cusips = [x["cusip"] for x in conn.execute(
        "SELECT cusip FROM issuer_cusips WHERE issuer_id = ? ORDER BY added_at, cusip", (issuer_id,))]
    return {
        "issuer_id": r["id"],
        "name": r["name"],
        "saved_at": r["saved_at"],
        "seconds": r["seconds"],
        "report": json.loads(r["report_json"]),
        "bbg": json.loads(r["bbg_json"] or "{}"),
        "cusips": cusips,
    }


def save_report(cusip: str, bbg: dict, report: dict, seconds: float) -> dict:
    """Save as the issuer's report, replacing any older one (no version history)."""
    iss  = report.get("issuer") or {}
    name = iss.get("name") or borrower_name(bbg) or bbg.get("ISSUER") or cusip
    key  = issuer_key(bbg) or norm(name)
    ts   = now_iso()
    vals = (name, iss.get("state") or bbg.get("STATE_CODE"), iss.get("sector"),
            json.dumps(report), json.dumps(bbg), round(seconds, 1), ts)
    with db() as conn:
        iid = find_issuer_by_cusip(conn, cusip) or find_issuer_by_key(conn, key)
        if iid:
            conn.execute("UPDATE issuers SET name=?, state=?, sector=?, report_json=?, bbg_json=?, seconds=?, saved_at=? "
                         "WHERE id=?", vals + (iid,))
        else:
            cur = conn.execute("INSERT INTO issuers (name, state, sector, report_json, bbg_json, seconds, saved_at, issuer_key) "
                               "VALUES (?,?,?,?,?,?,?,?)", vals + (key,))
            iid = cur.lastrowid
        link_cusip(conn, iid, cusip)
        return issuer_payload(conn, iid)


# ── Web enrichment ────────────────────────────────────────────────────────────

async def brave_search(client: httpx.AsyncClient, query: str, count: int = 4) -> list:
    if not BRAVE_API_KEY:
        return []
    try:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
            params={"q": query, "count": count, "text_decorations": False},
            timeout=8.0,
        )
        if resp.status_code != 200:
            return []
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
                for r in resp.json().get("web", {}).get("results", [])]
    except Exception:
        return []


async def emma_trade_data(client: httpx.AsyncClient, cusip: str) -> str:
    try:
        url = (f"https://emma.msrb.org/api/TradeActivity/GetTradeActivityByCusip?cusip={cusip}"
               "&startDate=&endDate=&sortField=tradeDate&sortOrder=desc&page=1&pageSize=10")
        resp = await client.get(url, timeout=8.0, follow_redirects=True)
        if resp.status_code != 200:
            return ""
        trades = resp.json().get("TradeActivityResult", {}).get("TradeActivityItems", [])
        if not trades:
            return ""
        lines = ["Recent MSRB EMMA trade activity:"]
        for t in trades[:8]:
            lines.append(f"  {t.get('TradeDate', '')[:10]} | {t.get('BuySellIndicator', '')} | "
                         f"Par: {t.get('ParTraded', 'N/A')} | Price: {t.get('Price', 'N/A')} | Yield: {t.get('Yield', 'N/A')}%")
        return "\n".join(lines)
    except Exception:
        return ""


def format_search_results(results: list, label: str) -> str:
    if not results:
        return ""
    lines = [f"{label}:"]
    for r in results:
        lines.append(f"  • {r['title']}")
        if r["snippet"]:
            lines.append(f"    {r['snippet'][:240]}")
        lines.append(f"    Source: {r['url']}")
    return "\n".join(lines)


async def gather_web_data(cusip: str, issuer_name: str) -> str:
    """All web lookups in parallel, hard cap 10s. Searches by the Bloomberg issuer name."""
    name = issuer_name or cusip
    year = datetime.now().year

    async def _all():
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=4.0)) as client:
            return await asyncio.gather(
                brave_search(client, f'"{name}" rating action Moody\'s S&P Fitch {year - 1} {year}'),
                brave_search(client, f'"{name}" municipal bond news {year}'),
                brave_search(client, f'"{name}" annual comprehensive financial report budget audit'),
                brave_search(client, f'"{name}" population economy largest employers taxpayers', 3),
                emma_trade_data(client, cusip),
                return_exceptions=True,
            )

    try:
        res = await asyncio.wait_for(_all(), timeout=10.0)
    except asyncio.TimeoutError:
        return ""
    res = [r if not isinstance(r, Exception) else ([] if i < 4 else "") for i, r in enumerate(res)]
    ratings, news, financials, economy, emma = res
    sections = [s for s in [
        emma,
        format_search_results(ratings,    "Web: rating actions"),
        format_search_results(news,       "Web: news & developments"),
        format_search_results(financials, "Web: financial reports"),
        format_search_results(economy,    "Web: economy & demographics"),
    ] if s]
    return ("## Live Web Data (retrieved now)\n" + "\n\n".join(sections)) if sections else ""


# ── Report generation: six analyst passes run in parallel ────────────────────
# Splitting the report into six simultaneous Claude calls keeps a full,
# detailed report near ~30-40 seconds. The page shows each section as it lands.

SYSTEM_PROMPT = """You are a senior municipal credit analyst at The Tribeca Group, a fixed income portfolio management team at Morgan Stanley. You write rigorous, specific, opinionated issuer-level credit research for portfolio managers.

CRITICAL RULE — NEVER GUESS THE ISSUER OR RATINGS: issuer identity and current ratings come from the ## Bloomberg Data section. Use ONLY that data for the issuer name and current ratings. Do not override or second-guess it. If an ultimate borrower/obligor is given (conduit deal), the credit is the BORROWER — analyze the borrower and name the conduit issuer separately.

Ground rules:
- This is ISSUER-LEVEL CREDIT research. Ignore bond-specific coupon, maturity, price and duration.
- Use the live web data for anything 2024 or later. Use your expertise for sector context, financial analysis and judgment.
- Be specific to this issuer: real figures, real names, real places. No generic muni boilerplate. If a figure is uncertain or estimated, say so (e.g. "~$1.2B (FY2024, est.)").
- In free-text fields use **bold** for key figures and conclusions and *italics* for emphasis. No emojis.
- Be complete but economical: every sentence should carry a fact or a judgment. No filler.
- Return ONLY valid JSON matching the schema. No preamble, no markdown fences."""

PART_PROMPTS = {
    "overview": """Produce the ISSUER OVERVIEW. JSON schema:
{
  "issuer": {
    "name": "Full official name of the credit (the borrower/obligor on conduit deals)",
    "conduit_issuer": "Conduit issuer name if applicable, else null",
    "sector": "GO | Water & Sewer | Electric | School District | Transportation | Healthcare | Higher Ed | Housing | Special Tax | Airport | Toll Road | Public Power | Other",
    "bond_type": "General Obligation | Revenue | COP/Lease | Special Assessment | Appropriation | Other",
    "state": "Two-letter state code",
    "description": "4-5 sentences: what the entity is, what it does, its size, and its credit profile in plain terms"
  },
  "service_area": {
    "summary": "Geographic area served or taxed — concrete (counties, cities, square miles)",
    "component_entities": ["For county / regional / district credits: every city, town, village or member entity inside it (names only). For utilities or authorities: member systems or service territories. Empty list if not applicable."],
    "population": "Population with year and trend (e.g. '1.2M (2023), +0.4%/yr')",
    "economy": "2-3 sentences on the economic base, wealth and income vs state/US, unemployment, key industries",
    "top_employers": ["Up to 6 largest employers (Name — approx. employees)"],
    "top_taxpayers_or_customers": ["Up to 6 largest taxpayers (tax-backed) or customers (revenue credits), with % concentration if known"]
  }
}""",

    "operations": """Produce the OPERATIONS, REVENUE and SECURITY section. JSON schema:
{
  "operations": {
    "governance": "2-3 sentences: governing body, how chosen, key officials, management quality",
    "summary": "3-4 sentences on operations: services, scale (customers, enrollment, beds, passengers, miles, etc.), capital plan, labor",
    "key_facts": ["5-8 short factual bullets: fiscal year end, budget size, employees, system size, rate-setting authority, etc."]
  },
  "revenue_sources": [ { "source": "Revenue source", "share": "~XX%", "notes": "One line: stability, pledging, trend" } ],
  "security_pledge": "3-4 sentences: exactly what secures the bonds — tax levy (limited/unlimited), net revenue pledge, rate covenant, reserve funds, additional bonds test, intercepts, appropriation risk, lien priority."
}""",

    "metrics": """Produce the KEY FINANCIAL METRICS. JSON schema:
{
  "key_metrics": [
    { "metric": "Sector-specific metric", "value": "Issuer value with units and fiscal year", "median": "Rating-category / sector median", "vs_median": "above | below | in-line", "favorable": true, "notes": "One short line of context" }
  ]
}
Give 10-12 metrics chosen for THIS sector (GO: full value per capita, fund balance %, debt/full value, debt service % of spend, pension/OPEB burden, tax base growth, income vs US. Utilities: DSCR, days cash, debt/plant, rates vs peers, customer concentration. Healthcare: operating margin, days cash, cushion, debt/cap, payer mix, market share. Higher ed: enrollment, demand ratios, endowment/debt, tuition dependence. Transportation/airports: enplanements, CPE, coverage, liquidity). "favorable" = true if credit-positive vs median.""",

    "ratings": """Produce the RATINGS, RECENT DEVELOPMENTS and SECTOR section. JSON schema:
{
  "ratings": {
    "moodys": { "rating": "per Bloomberg, or NR", "outlook": "Stable | Positive | Negative | Under Review | NR", "last_action": "One sentence: what the agency last did and why", "last_action_date": "Mon YYYY or null" },
    "sp":     { "rating": "...", "outlook": "...", "last_action": "...", "last_action_date": "..." },
    "fitch":  { "rating": "...", "outlook": "...", "last_action": "...", "last_action_date": "..." }
  },
  "rating_history": [ { "date": "Mon YYYY", "agency": "Moody's | S&P | Fitch | KBRA", "action": "Upgrade | Downgrade | Affirmed | Outlook change | Assigned | Watch", "detail": "One sentence: from -> to, and the stated reason" } ],
  "recent_developments": [ { "date": "Mon YYYY", "headline": "Short factual headline", "impact": "positive | negative | neutral", "detail": "One sentence with specifics" } ],
  "sector": {
    "name": "Sector name",
    "outlook": "Positive | Stable | Negative",
    "summary": "3 sentences on the national backdrop for this type of credit right now and what is driving it",
    "issuer_vs_sector": "1-2 sentences: how this issuer is positioned relative to the sector trend"
  }
}
Up to 5 rating history entries and up to 5 recent developments, most recent first.""",

    "credit": """Produce the KEY STRENGTHS and KEY RISKS section. JSON schema:
{
  "strengths": [ { "title": "Short headline", "detail": "2 sentences with specific figures" } ],
  "risks":     [ { "title": "Short headline", "detail": "2 sentences with specific figures" } ]
}
4-5 strengths and 4-5 risks.""",

    "view": """Produce the TRIBECA AI ANALYST VIEW. JSON schema:
{
  "credit_horizon": {
    "max_years": 20,
    "label": "Short (0-5y) | Intermediate (5-10y) | Long (10-20y) | Any tenor (20y+)",
    "rationale": "2 sentences: how far out this CREDIT holds up given its trajectory. Credit risk only — not duration or rate risk."
  },
  "investability": {
    "1yr":  { "verdict": "INVESTABLE | CAUTIOUS | AVOID", "rationale": "One short sentence, credit-based" },
    "5yr":  { "verdict": "...", "rationale": "..." },
    "10yr": { "verdict": "...", "rationale": "..." },
    "15yr": { "verdict": "...", "rationale": "..." },
    "20yr": { "verdict": "...", "rationale": "..." }
  },
  "overall_recommendation": "BUY | HOLD | REDUCE | AVOID",
  "analyst_summary": "5-6 sentences: the credit story, the single most important thing to know right now, how it compares to peers, and the stance. Direct and opinionated.",
  "what_would_change_view": ["3-5 specific triggers that would move the view up or down"],
  "data_note": "One sentence: which sources were used (Bloomberg, live web, EMMA) and anything that could not be verified"
}
max_years must match the label, and the investability verdicts must be consistent with credit_horizon.""",
}

PART_MAX_TOKENS = {p: 3000 for p in PART_PROMPTS}


def extract_json(text: str) -> dict:
    m = re.search(r"\{[\s\S]*\}", text.strip())
    if not m:
        raise ValueError("no JSON in response")
    return json.loads(m.group())


async def run_part(client: httpx.AsyncClient, part: str, context: str, model: str) -> dict:
    resp = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={
            "model": model,
            "max_tokens": PART_MAX_TOKENS[part],
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": f"{context}\n\n---\n\n{PART_PROMPTS[part]}"}],
        },
    )
    if resp.status_code != 200:
        try:
            msg = resp.json().get("error", {}).get("message", "")
        except Exception:
            msg = resp.text[:200]
        raise RuntimeError(f"Claude API error {resp.status_code}: {msg}")
    data = resp.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    try:
        return extract_json(text)
    except Exception as e:
        if data.get("stop_reason") == "max_tokens":
            raise RuntimeError(f"{part} section ran too long and was cut off")
        raise RuntimeError(f"could not read {part} section ({e})")


# ── Routes ────────────────────────────────────────────────────────────────────

class CusipRequest(BaseModel):
    cusip: str
    model: Optional[str] = None


@app.get("/")
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "MuniRadar Credit Intel API",
        "web_enrichment": bool(BRAVE_API_KEY),
        "bloomberg": "required",
    }


@app.post("/api/lookup")
async def lookup(req: CusipRequest):
    """Step 1 of a search: is this CUSIP's issuer already in the library?"""
    cusip = req.cusip.strip().upper()
    with db() as conn:
        iid = find_issuer_by_cusip(conn, cusip)
    bbg = {}
    if not iid:
        bbg = await get_bloomberg_cached(cusip)
        with db() as conn:
            iid = find_issuer_by_key(conn, issuer_key(bbg))
    if not iid:
        return {"found": False, "issuer": borrower_name(bbg) or bbg.get("ISSUER", "")}
    with db() as conn:
        link_cusip(conn, iid, cusip)
        p = issuer_payload(conn, iid)
    return {"found": True, "issuer_id": iid, "name": p["name"], "saved_at": p["saved_at"]}


@app.post("/api/report")
async def report(req: CusipRequest):
    """Step 2 (Pull new): Bloomberg -> web -> 6 parallel Claude passes, streamed as NDJSON."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")
    cusip = req.cusip.strip().upper()
    model = req.model or DEFAULT_MODEL
    t0 = time.time()
    bbg = await get_bloomberg_cached(cusip)

    async def stream():
        yield json.dumps({"event": "status", "message": "Gathering live web data…"}) + "\n"
        web = await gather_web_data(cusip, borrower_name(bbg) or bbg.get("ISSUER", ""))
        retrieved = (f"## Data actually retrieved for this report\n"
                     f"  Bloomberg: yes\n"
                     f"  Live web search results: {'yes' if 'Web:' in web else 'NONE — rely on your own knowledge and flag figures that may be stale'}\n"
                     f"  MSRB EMMA trade data: {'yes' if 'EMMA' in web else 'none'}\n"
                     f"In data_note, name ONLY the sources marked yes above.")
        context = f"Analyze the issuer behind CUSIP {cusip}.\n\n{format_bloomberg_data(bbg)}\n\n{web}\n\n{retrieved}"
        yield json.dumps({"event": "status", "message": "Writing report…"}) + "\n"

        report_data, errors = {}, []
        async with httpx.AsyncClient(timeout=httpx.Timeout(150.0, connect=10.0)) as client:
            tasks = {asyncio.create_task(run_part(client, p, context, model)): p for p in PART_PROMPTS}
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    part = tasks[t]
                    try:
                        data = t.result()
                        report_data.update(data)
                        yield json.dumps({"event": "part", "part": part, "data": data,
                                          "seconds": round(time.time() - t0, 1)}) + "\n"
                    except Exception as e:
                        errors.append(f"{part}: {e}")
                        yield json.dumps({"event": "part_error", "part": part, "message": str(e)}) + "\n"

        seconds = time.time() - t0
        if errors:
            yield json.dumps({"event": "error", "message": "Report incomplete — not saved. " + "; ".join(errors)}) + "\n"
            return
        try:
            saved = save_report(cusip, bbg, report_data, seconds)
            saved.pop("report", None)
            yield json.dumps({"event": "saved", "issuer": saved, "seconds": round(seconds, 1)}) + "\n"
        except Exception as e:
            yield json.dumps({"event": "error", "message": f"Report built but could not be saved: {e}"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/issuers")
def list_issuers():
    with db() as conn:
        rows = conn.execute("""
            SELECT i.id, i.name, i.state, i.sector, i.saved_at, i.report_json,
                   (SELECT GROUP_CONCAT(cusip, ',') FROM issuer_cusips c WHERE c.issuer_id = i.id) AS cusips
            FROM issuers i ORDER BY i.saved_at DESC, i.id DESC
        """).fetchall()
    out = []
    for r in rows:
        rep = json.loads(r["report_json"] or "{}")
        out.append({
            "id": r["id"], "name": r["name"], "state": r["state"], "sector": r["sector"],
            "saved_at": r["saved_at"],
            "cusips": r["cusips"].split(",") if r["cusips"] else [],
            "recommendation": rep.get("overall_recommendation"),
            "sector_outlook": (rep.get("sector") or {}).get("outlook"),
        })
    return out


@app.get("/api/issuers/{issuer_id}")
def get_issuer(issuer_id: int):
    with db() as conn:
        p = issuer_payload(conn, issuer_id)
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    return p


@app.delete("/api/issuers/{issuer_id}")
def delete_issuer(issuer_id: int):
    with db() as conn:
        cur = conn.execute("DELETE FROM issuers WHERE id = ?", (issuer_id,))
        conn.execute("DELETE FROM issuer_cusips WHERE issuer_id = ?", (issuer_id,))
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"deleted": issuer_id}
