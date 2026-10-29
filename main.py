import os
import json
import re
import asyncio
import html
import sqlite3
import time
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
DEFAULT_MODEL     = os.environ.get("DEFAULT_MODEL", "claude-sonnet-5-5")
BACKEND_VERSION   = "2026-10-29g (CUSIP web lookup for exact system/obligor)"
NAME_MODEL        = os.environ.get("NAME_MODEL", "claude-haiku-5-5")   # fast model for issuer-name expansion
DB_PATH           = os.environ.get("MUNIRADAR_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "muniradar.db"))

SYSTEM_PROMPT = """You are a senior municipal credit analyst at The Tribeca Group, a professional fixed income portfolio management team at Morgan Stanley. Your job is to produce rigorous, opinionated credit intelligence cards for muni bond CUSIPs. You have deep expertise in municipal credit structures, revenue pledge analysis, intercept mechanisms, sector benchmarks, and investability frameworks.

CRITICAL RULE — NEVER GUESS THE ISSUER OR RATINGS: All issuer identity data is provided from verified sources (Bloomberg or EMMA) in the ## Verified Bond Data section. Use ONLY that data for issuer name. For ratings, use Bloomberg data if provided; otherwise use ratings found in live web data. Do NOT invent or guess issuer names or ratings. If ratings are not confirmed, mark them NR and note the source uncertainty.

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
    "1yr":  { "verdict": "INVESTABLE | CAUTIOUS | AVOID", "rationale": "One sentence: is this ISSUER's credit safe to own for the next 1 year, and why" },
    "5yr":  { "verdict": "INVESTABLE | CAUTIOUS | AVOID", "rationale": "One sentence: is this issuer's credit safe to own out to 5 years, and why" },
    "10yr": { "verdict": "INVESTABLE | CAUTIOUS | AVOID", "rationale": "One sentence: is this issuer's credit safe to own out to 10 years, and why" },
    "15yr": { "verdict": "INVESTABLE | CAUTIOUS | AVOID", "rationale": "One sentence: is this issuer's credit safe to own out to 15 years, and why" },
    "20yr": { "verdict": "INVESTABLE | CAUTIOUS | AVOID", "rationale": "One sentence: is this issuer's credit safe to own out to 20 years, and why" }
  },
  "overall_recommendation": "ONE WORD ONLY: BUY | HOLD | REDUCE | AVOID (the stance on the issuer's credit)",
  "analyst_note": "2-3 sentence Tribeca AI Analyst take on the ISSUER's credit (not this specific bond): the single most important thing to know about this credit right now, and what would change the view. Be direct and opinionated.",
  "data_note": "Brief disclosure about data currency and sources used in this analysis — specify whether Bloomberg or EMMA was used for issuer identification, and how ratings were sourced"
}

This is ISSUER-LEVEL credit research: the same card is reused for every CUSIP from this issuer, so never let a single bond's maturity, call, coupon or price drive any judgment. Be specific to the actual issuer. Use real figures where known. Do not give generic muni boilerplate. If data is uncertain, say so in the notes fields rather than omitting or fabricating. Distinguish between bond types within an issuer (e.g. LADWP Power vs Water; LAUSD GO vs COPs). When live web data is provided in the prompt, prioritize it over your training data for recent developments."""


# ── Bloomberg data fetch (synchronous, runs in thread pool) ──────────────────

BLP_FIELDS = [
    "ISSUER",
    "SECURITY_DES",
    "RTG_MOODY",
    "RTG_SP",
    "RTG_FITCH",
    "MOODY_OUTLOOK",
    "SP_OUTLOOK",
    "FITCH_OUTLOOK",
    "RTG_MDY_WATCH",
    "RTG_SP_WATCH",
    "NXT_CALL_DT",
    "NXT_CALL_PX",
    "MATURITY",
    "CPN",
    "YLD_YTM_MID",
    "PX_MID",
    "Z_SPRD_MID",
    "OAS_SPREAD_MID",
    "TAX_STATUS",
    "STATE_CODE",
    "COLLAT_TYP",
    "ISSUE_DT",
    "AMT_OUTSTANDING",
    "DEBT_SERVICE_COVERAGE",
    "DAYS_CASH_ON_HAND",
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
    lines = ["## Verified Bond Data — Bloomberg (use as ground truth for issuer and ratings)"]
    label_map = {
        "ISSUER":               "Issuer name",
        "SECURITY_DES":         "Security description",
        "RTG_MOODY":            "Moody's rating",
        "RTG_SP":               "S&P rating",
        "RTG_FITCH":            "Fitch rating",
        "MOODY_OUTLOOK":        "Moody's outlook",
        "SP_OUTLOOK":           "S&P outlook",
        "FITCH_OUTLOOK":        "Fitch outlook",
        "RTG_MDY_WATCH":        "Moody's watch status",
        "RTG_SP_WATCH":         "S&P watch status",
        "NXT_CALL_DT":          "Next call date",
        "NXT_CALL_PX":          "Next call price",
        "MATURITY":             "Maturity date",
        "CPN":                  "Coupon",
        "YLD_YTM_MID":          "Yield to maturity (mid)",
        "PX_MID":               "Price (mid)",
        "Z_SPRD_MID":           "Z-spread (mid)",
        "OAS_SPREAD_MID":       "OAS spread (mid)",
        "TAX_STATUS":           "Tax status",
        "STATE_CODE":           "State",
        "COLLAT_TYP":           "Collateral / security type",
        "ISSUE_DT":             "Issue date",
        "AMT_OUTSTANDING":      "Amount outstanding",
        "DEBT_SERVICE_COVERAGE":"Debt service coverage ratio",
        "DAYS_CASH_ON_HAND":    "Days cash on hand",
    }
    for field, label in label_map.items():
        if field in data:
            lines.append(f"  {label}: {data[field]}")
    return "\n".join(lines)


# ── EMMA fallback — issuer name lookup ───────────────────────────────────────

async def emma_issuer_lookup(client: httpx.AsyncClient, cusip: str) -> str:
    """Try to get verified issuer name from EMMA bond detail."""
    try:
        url = f"https://emma.msrb.org/api/SecuritiesSearch/GetBondDetail?cusip={cusip}"
        resp = await client.get(url, timeout=10.0, follow_redirects=True)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        # Try common EMMA response fields for issuer name
        for key in ("IssuerName", "issuerName", "Issuer", "issuer", "ObligorName", "obligorName"):
            val = data.get(key, "")
            if val and isinstance(val, str) and len(val) > 3:
                return val.strip()
        # Try nested structures
        bond = data.get("BondDetail") or data.get("bondDetail") or data
        if isinstance(bond, dict):
            for key in ("IssuerName", "issuerName", "Issuer", "issuer"):
                val = bond.get(key, "")
                if val and isinstance(val, str) and len(val) > 3:
                    return val.strip()
        return ""
    except Exception:
        return ""


async def emma_security_search(client: httpx.AsyncClient, cusip: str) -> str:
    """Secondary EMMA endpoint for issuer name."""
    try:
        url = f"https://emma.msrb.org/api/SecuritiesSearch/GetSecurityDetails?cusip={cusip}"
        resp = await client.get(url, timeout=10.0, follow_redirects=True)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        for key in ("IssuerName", "issuerName", "ObligorName", "obligorName", "Issuer", "issuer"):
            val = data.get(key, "")
            if val and isinstance(val, str) and len(val) > 3:
                return val.strip()
        return ""
    except Exception:
        return ""


async def get_issuer_from_emma(cusip: str) -> str:
    """Try multiple EMMA endpoints to get verified issuer name."""
    async with httpx.AsyncClient(timeout=12.0) as client:
        results = await asyncio.gather(
            emma_issuer_lookup(client, cusip),
            emma_security_search(client, cusip),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, str) and r:
                return r
    return ""


def format_emma_data(issuer_name: str, cusip: str) -> str:
    """Format EMMA-sourced issuer data as a prompt section."""
    lines = ["## Verified Bond Data — EMMA (issuer name verified; ratings from web search)"]
    lines.append(f"  Issuer name: {issuer_name}")
    lines.append(f"  CUSIP: {cusip}")
    lines.append("  Note: Bloomberg unavailable. Ratings sourced from live web data below — treat as indicative.")
    return "\n".join(lines)


# ── OpenFIGI fallback — Bloomberg's free public identifier service ───────────
# No Terminal needed. Returns Bloomberg's own name/ticker for the CUSIP.

OPENFIGI_API_KEY = os.environ.get("OPENFIGI_API_KEY", "")  # optional; raises rate limit


async def get_issuer_from_openfigi(cusip: str) -> dict:
    """Look up the CUSIP on OpenFIGI. Returns {} if not found or unreachable."""
    headers = {"Content-Type": "application/json"}
    if OPENFIGI_API_KEY:
        headers["X-OPENFIGI-APIKEY"] = OPENFIGI_API_KEY
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.openfigi.com/v3/mapping",
                headers=headers,
                json=[{"idType": "ID_CUSIP", "idValue": cusip}],
            )
        if resp.status_code != 200:
            return {}
        result = resp.json()
        items = (result[0] or {}).get("data") if isinstance(result, list) and result else None
        if not items:
            return {}
        item = items[0]
        if not item.get("name"):
            return {}
        return {
            "name":         item.get("name", ""),
            "ticker":       item.get("ticker", ""),
            "security_type": item.get("securityType", ""),
            "market_sector": item.get("marketSector", ""),
            "figi":         item.get("figi", ""),
        }
    except Exception:
        return {}


def openfigi_search_name(name: str) -> str:
    """'NEW YORK CITY-D' -> 'NEW YORK CITY' (drop Bloomberg's series suffix for web searches)."""
    return re.sub(r"-[A-Z0-9]{1,3}$", "", (name or "").strip())


def format_openfigi_data(figi: dict, cusip: str) -> str:
    """Format OpenFIGI-sourced issuer data as a prompt section."""
    lines = ["## Verified Bond Data — Bloomberg OpenFIGI (issuer identity verified; ratings from web search)"]
    lines.append(f"  Bloomberg issuer short name: {figi['name']}")
    if figi.get("ticker"):
        lines.append(f"  Bloomberg ticker: {figi['ticker']}")
    if figi.get("market_sector"):
        lines.append(f"  Market sector: {figi['market_sector']}")
    if figi.get("security_type"):
        lines.append(f"  Security type: {figi['security_type']}")
    lines.append(f"  CUSIP: {cusip}")
    lines.append("  Note: Bloomberg's short name may be abbreviated and may carry a series suffix (e.g. '-D'). "
                 "Expand it to the issuer's full official name, but do not change WHICH entity it is.")
    lines.append("  Note: Bloomberg Terminal unavailable. Ratings sourced from live web data below — treat as indicative.")
    return "\n".join(lines)


# ── Web enrichment helpers ────────────────────────────────────────────────────

async def brave_search(client: httpx.AsyncClient, query: str, count: int = 5,
                       freshness: Optional[str] = None) -> list[dict]:
    """Brave web search. freshness='py' limits results to the past year."""
    if not BRAVE_API_KEY:
        return []
    params = {"q": query, "count": count, "text_decorations": False, "extra_snippets": True}
    if freshness:
        params["freshness"] = freshness
    try:
        # Brave's free plan allows ~1 search/second: retry when rate-limited (429)
        for attempt in range(8):
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
                params=params,
                timeout=10.0,
            )
            if resp.status_code != 429:
                break
            await asyncio.sleep(1.1 + 0.4 * attempt)
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = []
        def clean(t: str) -> str:   # strip <strong> tags and &#x27; style entities
            return html.unescape(re.sub(r"<[^>]+>", "", t or ""))
        for r in data.get("web", {}).get("results", []):
            results.append({
                "title":   clean(r.get("title", "")),
                "url":     r.get("url", ""),
                "snippet": clean(r.get("description", "")),
                "extra":   [clean(x) for x in (r.get("extra_snippets", []) or [])],
                "date":    r.get("page_age", "") or r.get("age", ""),
            })
        return results
    except Exception:
        return []


# Bloomberg/OpenFIGI names are abbreviated ("DALLAS-FORT WORTH TX INTL ARPT").
# News articles use plain English, so expand common abbreviations before searching.
_ABBREV = {
    "ARPT": "Airport", "INTL": "International", "AUTH": "Authority", "AGY": "Agency",
    "DEV": "Development", "FIN": "Finance", "FING": "Financing", "HSG": "Housing",
    "UNIV": "University", "SCH": "School", "DIST": "District", "CNTY": "County",
    "CO": "County", "TWP": "Township", "WTR": "Water", "SWR": "Sewer", "PWR": "Power",
    "ELEC": "Electric", "UTIL": "Utility", "UTILS": "Utilities", "TRANSN": "Transportation",
    "TRAN": "Transit", "TPK": "Turnpike", "HOSP": "Hospital", "MED": "Medical", "CTR": "Center",
    "HLTH": "Health", "FACS": "Facilities", "FAC": "Facility", "DORM": "Dormitory",
    "COMWLTH": "Commonwealth", "MUN": "Municipal", "MUNI": "Municipal", "IMPT": "Improvement",
    "REDEV": "Redevelopment", "INDL": "Industrial", "ENVIRONMENTAL": "Environmental",
    "SVCS": "Services", "SYS": "System", "REGL": "Regional", "MET": "Metropolitan",
    "METRO": "Metropolitan", "PUB": "Public", "PK": "Park", "PKG": "Parking",
    "ST": "State", "DEPT": "Department", "CMNTY": "Community", "COLL": "College", "INDPT": "Independent", "ISD": "Independent School District",
    "USD": "Unified School District", "GO": "", "REV": "", "BDS": "", "REF": "", "SER": "",
}


_US_STATES = set("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA PR RI SC SD TN TX UT VT VA WA WV WI WY".split())


def plain_issuer_name(name: str) -> str:
    """'DALLAS-FORT WORTH TX INTL ARPT' -> 'Dallas-Fort Worth TX International Airport'."""
    words = []
    for w in re.split(r"\s+", (name or "").strip()):
        key = w.upper().strip(",.")
        if key in ("OF", "AND", "&", "THE", "FOR"):
            words.append(key.lower() if key != "&" else "&")
        elif key in _ABBREV:
            if _ABBREV[key]:
                words.append(_ABBREV[key])
        elif key in _US_STATES:
            words.append(key)                      # state code, keep as-is
        else:
            words.append(w.title())
    return " ".join(words).strip()


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
    seen = set()
    for r in results:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        dated = f" [{r['date'][:10]}]" if r.get("date") else ""
        lines.append(f"  • {r['title']}{dated}")
        text = " ".join([r["snippet"]] + list(r.get("extra") or [])).strip()
        if text:
            lines.append(f"    {text[:700]}")
        lines.append(f"    Source: {r['url']}")
    return "\n".join(lines)


async def gather_web_data(cusip: str, issuer_name: str, include_ratings_search: bool = False) -> str:
    """Run web enrichment in parallel, hard cap 25s (allows for rate-limit retries)."""
    async def _fetch_all():
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=4.0)) as client:
            # Bloomberg-style ALL-CAPS abbreviations get expanded; already-expanded names are used as-is
            if not issuer_name:
                search_name = cusip
            elif issuer_name.isupper():
                search_name = plain_issuer_name(issuer_name)
            else:
                search_name = issuer_name
            year          = datetime.now().year
            ratings_query = f"{search_name} Moody's S&P Fitch rating outlook upgrade downgrade affirmed"
            news_query    = f"{search_name} bonds news {year}"
            credit_query  = f"{search_name} annual comprehensive financial report budget {year - 1} {year}"
            # One search per agency (any date), so one agency's pages can't crowd out the others
            moodys_query  = f"{search_name} Moody's rating outlook"
            sp_query      = f"{search_name} S&P Global Ratings rating outlook"
            fitch_query   = f"{search_name} Fitch Ratings rating outlook"
            tasks = [
                brave_search(client, ratings_query, count=6 if include_ratings_search else 5, freshness="py"),
                brave_search(client, news_query,    count=5, freshness="py"),
                brave_search(client, credit_query,  count=4),
                emma_trade_data(client, cusip),
                brave_search(client, moodys_query,  count=4),
                brave_search(client, sp_query,      count=4),
                brave_search(client, fitch_query,   count=4),
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

    try:
        results = await asyncio.wait_for(_fetch_all(), timeout=25.0)
    except asyncio.TimeoutError:
        return ""

    ratings_results, news_results, credit_results, emma_trades, moodys_results, sp_results, fitch_results = [
        r if not isinstance(r, Exception) else ("" if i == 3 else [])
        for i, r in enumerate(results)
    ]

    sections = []
    if emma_trades:
        sections.append(emma_trades)

    for block in [
        format_search_results(ratings_results, "Recent web (past 12 months): rating actions & current ratings"),
        format_search_results(moodys_results,  "Web (any date): Moody's rating references"),
        format_search_results(sp_results,      "Web (any date): S&P rating references"),
        format_search_results(fitch_results,   "Web (any date): Fitch rating references"),
        format_search_results(news_results,    "Recent web (past 12 months): news & developments"),
        format_search_results(credit_results,  "Recent web: credit & financial"),
    ]:
        if block:
            sections.append(block)

    if not sections:
        return ""

    today = datetime.now().strftime("%B %d, %Y")
    return (f"\n\n## Live Web Data Retrieved at Request Time (today is {today})\n"
            "Items are dated where the source provides a date. Newer information here OVERRIDES your training data.\n"
            + "\n\n".join(sections) + "\n")


# ── Issuer identity: Bloomberg -> OpenFIGI -> EMMA ───────────────────────────

_IDENTITY_CACHE: dict = {}   # cusip -> (timestamp, identity); reused by lookup -> report


async def cusip_web_hints(cusip: str) -> str:
    """Search the CUSIP number itself. Pages listing a CUSIP usually show its full
    description (e.g. '...POWER SYSTEM REVENUE BONDS'), which tells apart issuers whose
    Bloomberg short names are identical (LADWP Power vs Water, airport vs city)."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=4.0)) as client:
        results = await brave_search(client, f'"{cusip}"', count=6)
    lines = []
    for r in results:
        text = " ".join([r["title"], r["snippet"]] + list(r.get("extra") or []))
        lines.append(f"- {text[:400]}")
    return "\n".join(lines)[:2500]


async def expand_issuer_name(short_name: str, ticker: str, cusip: str, hints: str = "") -> dict:
    """Bloomberg short names are abbreviated and ambiguous ('DALLAS & FORT WORTH-B').
    A quick call to a fast model turns short name + ticker into the full issuer/obligor
    name that news and rating agencies actually use. Returns {} on any failure."""
    if not ANTHROPIC_API_KEY or not short_name:
        return {}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={
                    "model": NAME_MODEL,
                    "max_tokens": 1500,
                    "system": ("You identify US municipal bond issuers from Bloomberg identifiers. Return ONLY JSON: "
                               '{"official_name": "...", "search_name": "...", "confirmed": true}. official_name = full '
                               "legal name of the obligor INCLUDING the specific system or credit that secures the bonds "
                               "(e.g. 'Department of Water and Power of the City of Los Angeles — Power System'). Never include a bond "
                               "series, year, coupon or issue name in either field — the same name must apply to every bond of that "
                               "obligor/system. "
                               "search_name = the name news articles and rating agencies use for that obligor/system. "
                               "Bloomberg short names are often truncated (e.g. 'LOS ANG DEPT OF WTR' = Dept of Water AND "
                               "Power, which has separate Power and Water systems) — do not infer the system from a "
                               "truncated name. Use the web results for this exact CUSIP to determine the specific "
                               "system/obligor, and the ticker to disambiguate (e.g. 'DALAPT' = Dallas/Fort Worth airport, "
                               "not the City of Dallas). Set confirmed=false if the specific system/obligor cannot be "
                               "determined from the evidence; never invent."),
                    "messages": [{"role": "user", "content":
                                  f"Bloomberg short name: {short_name}\nBloomberg ticker: {ticker}\nCUSIP: {cusip}\n\n"
                                  f"Web results for this exact CUSIP:\n{hints or '(none found)'}"}],
                },
            )
        if resp.status_code != 200:
            return {}
        text = "".join(b.get("text", "") for b in resp.json().get("content", []) if b.get("type") == "text")
        m = re.search(r"\{[\s\S]*\}", text)
        names = json.loads(m.group()) if m else {}
        def tidy(n: str) -> str:   # drop any series / year / issue detail that slipped in
            n = re.sub(r"\s*\([^)]*\)", "", str(n))
            n = re.sub(r",?\s*(\d{4}\s+)?Series\b.*$", "", n, flags=re.I)
            n = re.sub(r"\s+(Revenue|Refunding)\s+Bonds.*$", "", n, flags=re.I)
            n = re.sub(r"\s*[—-]\s*(Unlimited Tax |Limited Tax )?General Obligation( Bonds)?$", "", n, flags=re.I)  # GO credit = the entity itself
            return n.strip(" ,—-")
        out = {k: tidy(names[k]) for k in ("official_name", "search_name") if names.get(k)}
        out["confirmed"] = names.get("confirmed", True) is not False
        return out
    except Exception:
        return {}


async def resolve_identity(cusip: str) -> dict:
    """Identify the issuer behind a CUSIP from verified sources, in order of preference.
    Returns {data_source, issuer_name, section, blp_data}."""
    hit = _IDENTITY_CACHE.get(cusip)
    if hit and time.time() - hit[0] < 600:
        return hit[1]

    # ── Step 1: Try Bloomberg first ──
    blp_data = {}
    data_source = "bloomberg"
    bloomberg_section = ""
    issuer_name = ""

    try:
        blp_data = await asyncio.wait_for(fetch_bloomberg(cusip), timeout=10.0)
        if blp_data:
            bloomberg_section = format_bloomberg_data(blp_data)
            issuer_name = blp_data.get("ISSUER", "")
    except (asyncio.TimeoutError, RuntimeError):
        blp_data = {}
    names = {}
    if blp_data and issuer_name:
        names = await expand_issuer_name(issuer_name, blp_data.get("SECURITY_DES", ""), cusip,
                                         await cusip_web_hints(cusip))

    # ── Step 2: Fall back to OpenFIGI (Bloomberg's free identifier service), then EMMA ──
    if not blp_data or not issuer_name:
        figi = await get_issuer_from_openfigi(cusip)
        if figi:
            data_source = "openfigi"
            issuer_name = openfigi_search_name(figi["name"])
            names = await expand_issuer_name(figi["name"], figi.get("ticker", ""), cusip,
                                             await cusip_web_hints(cusip))
            bloomberg_section = format_openfigi_data(figi, cusip)
        else:
            data_source = "emma"
            issuer_name = await get_issuer_from_emma(cusip)
            if issuer_name:
                bloomberg_section = format_emma_data(issuer_name, cusip)
            else:
                # No verified source — still proceed but warn Claude explicitly
                bloomberg_section = f"## Bond Identification\n  CUSIP: {cusip}\n  WARNING: Neither Bloomberg, OpenFIGI nor EMMA returned issuer data. Do NOT guess the issuer name. Set issuer name to 'Unknown — CUSIP {cusip}' and mark all ratings NR."
                data_source = "none"

    if names.get("official_name"):
        bloomberg_section += (f"\n  Expanded issuer name (from Bloomberg short name + ticker + CUSIP web lookup): {names['official_name']}"
                              f"\n  Commonly referred to as: {names.get('search_name', names['official_name'])}")
        if not names.get("confirmed", True):
            bloomberg_section += ("\n  WARNING: the specific system/obligor for this CUSIP could NOT be confirmed "
                                  "(e.g. Power vs Water system). Do not assume one: say so prominently in issuer.description "
                                  "and data_note, and keep the analysis to what is common to the issuer.")
    identity = {"data_source": data_source, "issuer_name": issuer_name,
                "official_name": names.get("official_name", ""),
                "search_name": names.get("search_name") or names.get("official_name") or "",
                "section": bloomberg_section, "blp_data": blp_data}
    if data_source != "none":
        _IDENTITY_CACHE[cusip] = (time.time(), identity)
    return identity


# ── Issuer library (SQLite) — one saved report per issuer ────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def normalize_rec(rec) -> str:
    """Keep just BUY / HOLD / REDUCE / AVOID even if the model wrote a sentence."""
    m = re.search(r"\b(BUY|HOLD|REDUCE|AVOID)\b", str(rec or "").upper())
    return m.group(1) if m else ""


def issuer_key(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (name or "").upper())


def find_issuer(conn, cusip: str, key: str = "") -> Optional[int]:
    row = conn.execute("SELECT issuer_id FROM issuer_cusips WHERE cusip = ?", (cusip,)).fetchone()
    if row:
        return row["issuer_id"]
    if key:
        row = conn.execute("SELECT id FROM issuers WHERE issuer_key = ?", (key,)).fetchone()
        if row:
            return row["id"]
    return None


def link_cusip(conn, issuer_id: int, cusip: str):
    conn.execute(
        "INSERT INTO issuer_cusips (cusip, issuer_id, added_at) VALUES (?,?,?) "
        "ON CONFLICT(cusip) DO UPDATE SET issuer_id = excluded.issuer_id",
        (cusip, issuer_id, now_iso()),
    )


def library_meta(conn, issuer_id: int) -> Optional[dict]:
    r = conn.execute("SELECT id, name, saved_at FROM issuers WHERE id = ?", (issuer_id,)).fetchone()
    if not r:
        return None
    cusips = [x["cusip"] for x in conn.execute(
        "SELECT cusip FROM issuer_cusips WHERE issuer_id = ? ORDER BY added_at, cusip", (issuer_id,))]
    return {"issuer_id": r["id"], "name": r["name"], "saved_at": r["saved_at"], "cusips": cusips}


def save_report(cusip: str, identity: dict, intel: dict) -> dict:
    """Save as the issuer's report, replacing any older one."""
    iss  = intel.get("issuer") or {}
    name = iss.get("name") or identity["issuer_name"] or cusip
    key  = issuer_key(identity.get("official_name") or identity["issuer_name"]) or issuer_key(name)
    vals = (name, iss.get("state"), iss.get("sector"), json.dumps(intel),
            json.dumps(identity.get("blp_data") or {}), now_iso())
    with db() as conn:
        iid = find_issuer(conn, cusip, key)
        if iid:
            conn.execute("UPDATE issuers SET name=?, state=?, sector=?, report_json=?, bbg_json=?, saved_at=? WHERE id=?",
                         vals + (iid,))
        else:
            cur = conn.execute("INSERT INTO issuers (name, state, sector, report_json, bbg_json, saved_at, issuer_key) "
                               "VALUES (?,?,?,?,?,?,?)", vals + (key,))
            iid = cur.lastrowid
        link_cusip(conn, iid, cusip)
        return library_meta(conn, iid)


# ── Request models ────────────────────────────────────────────────────────────

class CreditIntelRequest(BaseModel):
    cusip: str
    pdf_text: Optional[str] = None
    model: Optional[str] = None


class LookupRequest(BaseModel):
    cusip: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "MuniRadar Credit Intel API",
        "web_enrichment": bool(BRAVE_API_KEY),
        "bloomberg": "preferred (OpenFIGI, then EMMA fallback active)",
        "version": BACKEND_VERSION,
        "model": DEFAULT_MODEL,
    }


@app.post("/api/credit-intel")
async def credit_intel(req: CreditIntelRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    cusip = req.cusip.strip().upper()
    model = req.model or DEFAULT_MODEL

    # ── Steps 1-2: Bloomberg -> OpenFIGI -> EMMA ──
    identity = await resolve_identity(cusip)
    data_source       = identity["data_source"]
    issuer_name       = identity["issuer_name"]
    bloomberg_section = identity["section"]

    # ── Step 3: Web enrichment — use more ratings search if no Bloomberg ──
    include_ratings_search = (data_source != "bloomberg")
    web_data = await gather_web_data(cusip, identity.get("search_name") or issuer_name,
                                     include_ratings_search=include_ratings_search)

    pdf_section = ""
    if req.pdf_text:
        pdf_section = f"\n\n## Uploaded Research / Document\n{req.pdf_text[:12000]}\n"

    # ── Step 4: Build prompt ──
    if data_source == "bloomberg":
        data_instructions = """Generate a complete credit intelligence card using the Bloomberg data above as the authoritative source for issuer identity and ratings.
1. Use ONLY the Bloomberg issuer name — do not alter or guess it
2. Use EXACTLY the Bloomberg ratings and outlooks — do not override with training data
3. Revenue sources and security pledge — infer from security type and sector
4. Key financial metrics vs sector benchmarks (use Bloomberg DSCR/DCOH if provided)
5. Recent developments — prioritize live web data for anything 2024 or later
6. Investability is ISSUER-LEVEL: for each horizon (1, 5, 10, 15, 20 years), answer "is this issuer's credit safe to own out to that horizon?" based on its credit trajectory (finances, pensions, economy, structural balance, event risk). The same answer must apply to ANY CUSIP from this issuer — ignore this bond's own maturity, call date, coupon and price entirely
7. In data_note, confirm Bloomberg was primary source"""
    elif data_source == "openfigi":
        data_instructions = """Generate a complete credit intelligence card. The issuer above is verified by Bloomberg's OpenFIGI service — that is the entity to analyze. Expand Bloomberg's short name to the full official issuer name (e.g. 'NEW YORK CITY-D' -> 'City of New York'), but do not substitute a different entity. The Bloomberg Terminal was unavailable, so ratings must be sourced from the live web data provided below.
1. Analyze ONLY the OpenFIGI-identified issuer — do not guess a different one
2. For ratings: take them from the live web data. Use the MOST RECENT dated rating or rating action found for each agency (an outlook change or affirmation also confirms the current rating). Older sources are acceptable if nothing newer exists — put the source date in last_action (e.g. "Affirmed A1, outlook revised to positive — Mar 2026")
3. Mark an agency NR only if no web source mentions that agency's rating at all, and note web search was inconclusive
4. Revenue sources and security pledge — infer from security type and sector
5. Recent developments — use live web data
6. Investability is ISSUER-LEVEL: for each horizon (1, 5, 10, 15, 20 years), answer "is this issuer's credit safe to own out to that horizon?" based on its credit trajectory (finances, pensions, economy, structural balance, event risk). The same answer must apply to ANY CUSIP from this issuer — ignore this bond's own maturity, call date, coupon and price entirely
7. In data_note, note that the issuer was verified via Bloomberg OpenFIGI and ratings sourced from web search"""
    elif data_source == "emma":
        data_instructions = """Generate a complete credit intelligence card. The issuer name above is verified from EMMA — use it exactly. Bloomberg was unavailable so ratings must be sourced from the live web data provided below.
1. Use ONLY the EMMA-verified issuer name — do not alter or guess it
2. For ratings: take them from the live web data. Use the MOST RECENT dated rating or rating action found for each agency (an outlook change or affirmation also confirms the current rating). Older sources are acceptable if nothing newer exists — put the source date in last_action (e.g. "Affirmed A1, outlook revised to positive — Mar 2026")
3. Mark an agency NR only if no web source mentions that agency's rating at all, and note web search was inconclusive
4. Revenue sources and security pledge — infer from security type and sector
5. Recent developments — use live web data
6. Investability is ISSUER-LEVEL: for each horizon (1, 5, 10, 15, 20 years), answer "is this issuer's credit safe to own out to that horizon?" based on its credit trajectory (finances, pensions, economy, structural balance, event risk). The same answer must apply to ANY CUSIP from this issuer — ignore this bond's own maturity, call date, coupon and price entirely
7. In data_note, note that issuer was verified via EMMA and ratings sourced from web search"""
    else:
        data_instructions = """WARNING: No verified issuer data was available from Bloomberg or EMMA. Do NOT guess the issuer from the CUSIP digits.
1. Set issuer name to the CUSIP only — do not fabricate a name
2. Mark all ratings NR
3. Note in data_note that no verified source was available
4. Still provide whatever sector analysis and investability framework you can based on any web data found"""

    user_message = f"""Analyze CUSIP: {cusip}

{bloomberg_section}
{web_data}{pdf_section}

{data_instructions}"""

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 16000,   # room for the model's thinking + the full JSON card
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_message}],
            },
        )

    if response.status_code != 200:
        err = response.json().get("error", {})
        raise HTTPException(status_code=response.status_code, detail=err.get("message", "Anthropic API error"))

    data = response.json()
    # Newer models may return a "thinking" block before the answer — read only the text blocks
    raw  = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()

    match = re.search(r'\{[\s\S]*\}', raw)
    if not match:
        raise HTTPException(status_code=500, detail="Could not parse model response as JSON")

    try:
        intel = json.loads(match.group())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error: {e}")

    intel["overall_recommendation"] = normalize_rec(intel.get("overall_recommendation")) or intel.get("overall_recommendation")

    # ── Save to the issuer library (replaces any older report for this issuer) ──
    if data_source != "none":
        try:
            intel["library"] = save_report(cusip, identity, intel)
        except Exception:
            pass

    return intel


@app.post("/api/lookup")
async def lookup(req: LookupRequest):
    """Before running a new report: is this CUSIP's issuer already in the library?"""
    cusip = req.cusip.strip().upper()
    with db() as conn:
        iid = find_issuer(conn, cusip)
    if not iid:
        identity = await resolve_identity(cusip)
        if identity["data_source"] != "none":
            with db() as conn:
                iid = find_issuer(conn, cusip, issuer_key(identity.get("official_name") or identity["issuer_name"]))
    if not iid:
        return {"found": False}
    with db() as conn:
        link_cusip(conn, iid, cusip)
        meta = library_meta(conn, iid)
    return {"found": True, **meta}


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
            "recommendation": normalize_rec(rep.get("overall_recommendation")),
        })
    return out


@app.get("/api/issuers/{issuer_id}")
def get_issuer(issuer_id: int):
    with db() as conn:
        r = conn.execute("SELECT report_json FROM issuers WHERE id = ?", (issuer_id,)).fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="Not found")
        intel = json.loads(r["report_json"])
        intel["library"] = library_meta(conn, issuer_id)
    cusips = intel["library"]["cusips"]
    return {"cusip": cusips[0] if cusips else "", "intel": intel}


@app.delete("/api/issuers/{issuer_id}")
def delete_issuer(issuer_id: int):
    with db() as conn:
        cur = conn.execute("DELETE FROM issuers WHERE id = ?", (issuer_id,))
        conn.execute("DELETE FROM issuer_cusips WHERE issuer_id = ?", (issuer_id,))
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"deleted": issuer_id}
