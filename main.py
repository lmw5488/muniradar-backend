import os
import json
import re
import asyncio
import html
import sqlite3
import time
from datetime import datetime, timezone, timedelta
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
BACKEND_VERSION   = "2026-10-29j (portfolio scan)"
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
    { "metric": "Metric name", "value": "Value with units AND its fiscal year, e.g. '1.9x (FY2025)'", "vs_median": "above | below | in-line", "notes": "Context vs sector median; say so if this is not the latest fiscal year" }
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

KEY METRICS: use the MOST RECENT fiscal year available. The '## Full Text of Recent Reports' section (rating agency releases, financial statements) is the best source for current figures — prefer it over your training data. Always state the fiscal year of each figure.

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


_FIGI_CACHE: dict = {}   # cusip -> parsed OpenFIGI result ({} = not found)


def _parse_figi(entry) -> dict:
    items = (entry or {}).get("data") if isinstance(entry, dict) else None
    if not items or not items[0].get("name"):
        return {}
    item = items[0]
    return {
        "name":          item.get("name", ""),
        "ticker":        item.get("ticker", ""),
        "security_type": item.get("securityType", ""),
        "market_sector": item.get("marketSector", ""),
        "figi":          item.get("figi", ""),
    }


async def openfigi_batch(cusips: list) -> None:
    """Look up many CUSIPs on OpenFIGI (10 per request without a key, 100 with one) and cache them.
    Retries when OpenFIGI's rate limit (25 requests/min without a key) is hit."""
    todo = [c for c in dict.fromkeys(cusips) if c not in _FIGI_CACHE]
    size = 100 if OPENFIGI_API_KEY else 10
    headers = {"Content-Type": "application/json"}
    if OPENFIGI_API_KEY:
        headers["X-OPENFIGI-APIKEY"] = OPENFIGI_API_KEY
    async with httpx.AsyncClient(timeout=20.0) as client:
        for i in range(0, len(todo), size):
            chunk = todo[i:i + size]
            for attempt in range(6):
                try:
                    resp = await client.post("https://api.openfigi.com/v3/mapping", headers=headers,
                                             json=[{"idType": "ID_CUSIP", "idValue": c} for c in chunk])
                except Exception:
                    resp = None
                if resp is not None and resp.status_code == 200:
                    for c, entry in zip(chunk, resp.json()):
                        _FIGI_CACHE[c] = _parse_figi(entry)
                    break
                await asyncio.sleep(3 + 3 * attempt)


async def get_issuer_from_openfigi(cusip: str) -> dict:
    """Look up the CUSIP on OpenFIGI. Returns {} if not found or unreachable."""
    if cusip in _FIGI_CACHE:
        return _FIGI_CACHE[cusip]
    await openfigi_batch([cusip])
    return _FIGI_CACHE.get(cusip, {})


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

BRAVE_QPS    = float(os.environ.get("BRAVE_QPS", "1"))   # Brave free plan = 1 search/second; raise if on a paid plan
_BRAVE_LOCK  = asyncio.Lock()
_BRAVE_LAST  = [0.0]


async def _brave_pace():
    """Space Brave requests out so they don't get rejected for exceeding the plan's rate."""
    async with _BRAVE_LOCK:
        wait = _BRAVE_LAST[0] + (1.05 / BRAVE_QPS) - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        _BRAVE_LAST[0] = time.time()


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
            await _brave_pace()
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


# ── Full text of the most useful report pages ────────────────────────────────
# Snippets rarely contain the numbers. Rating-agency releases (Fitch, KBRA, S&P, Moody's)
# and issuer financial pages usually list the latest fiscal year's coverage, liquidity,
# leverage — so open the best few and give Claude the text.

_REPORT_DOMAIN_SCORE = {
    "fitchratings.com": 10, "kbra.com": 10, "spglobal.com": 9, "moodys.com": 9,
    "businesswire.com": 8, "prnewswire.com": 6, "bondbuyer.com": 5, "emma.msrb.org": 0,
}
_SKIP_DOMAINS = ("youtube.com", "linkedin.com", "facebook.com", "twitter.com", "x.com", "wikipedia.org")
_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/126.0 Safari/537.36"}


def _domain(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url or "")
    return (m.group(1).lower() if m else "").removeprefix("www.")


def pick_report_urls(*result_lists, limit: int = 3) -> list:
    scored, seen = [], set()
    for results in result_lists:
        for r in results or []:
            url, dom = r.get("url", ""), _domain(r.get("url", ""))
            if not url or url in seen or any(dom.endswith(d) for d in _SKIP_DOMAINS):
                continue
            seen.add(url)
            score = next((v for d, v in _REPORT_DOMAIN_SCORE.items() if dom.endswith(d)), None)
            if score is None:
                score = 7 if (dom.endswith(".gov") or dom.endswith(".org") or "investor" in url.lower()) else 3
            if score <= 0:
                continue
            scored.append((score, r.get("date") or "", url))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [u for _, _, u in scored[:limit]]


def _html_to_text(raw: str) -> str:
    # Many agency sites (e.g. Fitch) render from embedded JSON — pull long strings out of scripts too
    script_strings = []
    for js in re.findall(r"<script[^>]*>([\s\S]*?)</script>", raw, flags=re.I):
        for m in re.findall(r'"((?:[^"\\]|\\.){120,})"', js):
            try:
                script_strings.append(json.loads(f'"{m}"'))
            except Exception:
                pass
    body = re.sub(r"<(script|style|noscript|svg|nav|footer|header)[^>]*>[\s\S]*?</\1>", " ", raw, flags=re.I)
    body = re.sub(r"<br\s*/?>|</p>|</li>|</h\d>|</tr>", "\n", body, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", body))
    if len(text.split()) < 150 and script_strings:
        text += "\n" + "\n".join(html.unescape(re.sub(r"<[^>]+>", " ", s)) for s in script_strings)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


async def fetch_page_text(client: httpx.AsyncClient, url: str, max_chars: int = 7000) -> str:
    try:
        resp = await client.get(url, headers=_UA, follow_redirects=True, timeout=8.0)
        if resp.status_code != 200 or len(resp.content) > 6_000_000:
            return ""
        ctype = resp.headers.get("content-type", "").lower()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            try:
                import io
                from pypdf import PdfReader          # optional: pip install pypdf
                reader = PdfReader(io.BytesIO(resp.content))
                text = "\n".join((pg.extract_text() or "") for pg in reader.pages[:10])
            except Exception:
                return ""
        else:
            text = _html_to_text(resp.text)
        if len(text) < 300:
            return ""
        # Start near the substance (agency releases put the numbers under these headings)
        for marker in ("KEY RATING DRIVERS", "Key Rating Drivers", "RATING RATIONALE", "Rating Rationale",
                       "Credit Highlights", "CREDIT HIGHLIGHTS", "Financial Highlights", "FINANCIAL HIGHLIGHTS"):
            i = text.find(marker)
            if i > 0:
                text = text[max(0, i - 300):]
                break
        return text[:max_chars]
    except Exception:
        return ""


async def fetch_report_texts(urls: list) -> str:
    if not urls:
        return ""
    async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=4.0)) as client:
        texts = await asyncio.gather(*[fetch_page_text(client, u) for u in urls], return_exceptions=True)
    parts = [f"### Source: {u}\n{t}" for u, t in zip(urls, texts) if isinstance(t, str) and t]
    if not parts:
        return ""
    return ("\n\n## Full Text of Recent Reports (rating agency releases / financial reports — use for the latest "
            "fiscal-year metrics, ratings and outlooks)\n" + "\n\n".join(parts) + "\n")


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

    # Open the best few report pages and read their full text (hard cap 12s)
    try:
        reports = await asyncio.wait_for(
            fetch_report_texts(pick_report_urls(ratings_results, fitch_results, sp_results, moodys_results, credit_results)),
            timeout=12.0)
    except asyncio.TimeoutError:
        reports = ""

    today = datetime.now().strftime("%B %d, %Y")
    return (f"\n\n## Live Web Data Retrieved at Request Time (today is {today})\n"
            "Items are dated where the source provides a date. Newer information here OVERRIDES your training data.\n"
            + "\n\n".join(sections) + "\n" + reports)


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


# ── Portfolio scan ────────────────────────────────────────────────────────────
# Upload CUSIP + par -> identify every holding -> quick credit scan per issuer
# (saved library report reused when available) -> par-weighted stats -> AI view
# of how conservative the portfolio is and its top risks. Runs as a background
# job; the page polls for progress. Nothing is saved.

_JOBS: dict = {}
SCAN_CONCURRENCY = 5

_MDY = ["AAA", "AA1", "AA2", "AA3", "A1", "A2", "A3", "BAA1", "BAA2", "BAA3", "BA1", "BA2", "BA3",
        "B1", "B2", "B3", "CAA1", "CAA2", "CAA3", "CA", "C"]
_SPF = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
        "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"]


def cusip_valid(c: str) -> bool:
    c = (c or "").upper()
    if not re.fullmatch(r"[0-9A-Z*@#]{8}[0-9]", c):
        return False
    total = 0
    for i, ch in enumerate(c[:8]):
        v = int(ch) if ch.isdigit() else (ord(ch) - 55 if ch.isalpha() else {"*": 36, "@": 37, "#": 38}[ch])
        if i % 2:
            v *= 2
        total += v // 10 + v % 10
    return (10 - total % 10) % 10 == int(c[8])


def rating_notch(r) -> Optional[int]:
    """'Aa2' / 'AA-' / 'A1 (sf)' -> 1 (AAA) ... 22 (D). None for NR/WR/unknown."""
    t = re.sub(r"\(.*?\)|\*.*$|/.*$", "", str(r or "")).strip().upper().replace(" ", "")
    if not t or t in ("NR", "WR", "N/A", "NA"):
        return None
    if t in _MDY and not t.endswith(("+", "-")) and (t != "AAA" or str(r).strip().startswith("Aaa")):
        return _MDY.index(t) + 1
    if t in _SPF:
        return _SPF.index(t) + 1
    return _MDY.index(t) + 1 if t in _MDY else None


def rating_bucket(notch: Optional[int]) -> str:
    if notch is None:
        return "NR"
    return "AAA" if notch == 1 else "AA" if notch <= 4 else "A" if notch <= 7 else "BBB" if notch <= 10 else "Below IG"


async def claude_json(system: str, prompt: str, max_tokens: int = 6000, model: Optional[str] = None) -> dict:
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model or DEFAULT_MODEL, "max_tokens": max_tokens, "system": system,
                  "messages": [{"role": "user", "content": prompt}]},
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Claude API error {resp.status_code}")
    text = "".join(b.get("text", "") for b in resp.json().get("content", []) if b.get("type") == "text")
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise RuntimeError("Could not read model response")
    return json.loads(m.group())


SCAN_SYSTEM = """You are a conservative municipal credit analyst at The Tribeca Group (Morgan Stanley) screening an outside portfolio for CREDIT RISK. Be skeptical and specific. Issuer-level credit only — ignore coupon, maturity, price and duration.
Ratings: take them ONLY from the evidence provided (saved report or web results); use the most recent dated rating per agency; if an agency's rating is not in the evidence, use "NR". Never invent ratings.
Return ONLY JSON:
{
  "issuer": "Full official name of the obligor/system",
  "sector": "GO | Water & Sewer | Electric | School District | Transportation | Airport | Toll Road | Healthcare | Higher Ed | Housing | Special Tax | Land-Secured | Appropriation/Lease | Other",
  "state": "Two-letter code",
  "ratings": {"moodys": {"rating": "", "outlook": ""}, "sp": {"rating": "", "outlook": ""}, "fitch": {"rating": "", "outlook": ""}},
  "risk_score": 1,
  "downgrade_risk": 1,
  "spread_beta": "low | medium | high",
  "key_risk": "The single biggest credit concern, <= 12 words",
  "flags": ["short tags, e.g. negative outlook, below A, unrated, conduit, project finance, land-secured, appropriation risk, small private college, standalone hospital, pension stress, structural deficit, declining population, concentration in one employer/taxpayer, litigation, recent downgrade"]
}
risk_score: 1 = very low credit risk (e.g. AAA/AA essential-service, stable), 3 = moderate, 5 = high (spec-grade, distressed or highly speculative structure).
downgrade_risk: 1-5 likelihood of a downgrade in the next 12-24 months (negative outlooks/watches, deteriorating trends => higher).
spread_beta: how much this credit's spread would likely widen in a muni sell-off — high for lower-rated, negative-outlook, deteriorating or high-beta sectors (hospitals, higher ed, special tax, project finance), low for high-grade essential-service and strong GOs."""

PORTFOLIO_SYSTEM = """You are a conservative municipal credit portfolio manager at The Tribeca Group (Morgan Stanley) reviewing an OUTSIDE portfolio a prospect holds. Your job is to surface and emphasize its CREDIT RISKS clearly and specifically. Strengths are secondary. Issuer-level credit only — not duration or interest-rate risk.
Return ONLY JSON:
{
  "conservativeness": {"rank": 3, "label": "Very aggressive | Aggressive | Moderate | Conservative | Very conservative", "rationale": "2 sentences on why"},
  "summary": "2-3 sentence headline view of the portfolio's credit risk for a PM",
  "top_risks": [
    {"title": "2-4 word risk name (e.g. Downgrade risk, Spread risk, Lower-quality exposure, Issuer concentration, Sector headwinds, Structural risk, State concentration)",
     "severity": 4,
     "detail": "<= 25 words: what it is in this portfolio, with % of par where useful",
     "issuers": ["the issuers driving it"]}
  ],
  "strengths": ["<= 12 words each, at most 3"]
}
conservativeness.rank: 5 = very conservative (high-grade, diversified, stable), 1 = very aggressive. Severity: 1 = minor ... 5 = serious. Give exactly 5 top_risks, most severe first. Consider downgrade risk, spread risk (lower-rated, negative-outlook, deteriorating or high-beta credits will be more volatile), credit quality mix, unrated paper, issuer/sector/state concentration, sector headwinds and structural risks. Use the computed statistics — do not recompute them differently."""


def _report_context(report: dict) -> str:
    keep = {k: report.get(k) for k in ("issuer", "ratings", "risks", "strengths", "key_metrics", "recent_developments",
                                       "investability", "overall_recommendation", "analyst_note", "security_pledge")}
    return "## Saved MuniRadar report for this issuer\n" + json.dumps(keep)[:9000]


async def scan_issuer(group: dict) -> dict:
    name = group["official_name"] or group["issuer_name"]
    context = ""
    with db() as conn:
        iid = None
        for c in group["cusips"]:
            iid = find_issuer(conn, c)
            if iid:
                break
        iid = iid or find_issuer(conn, "", issuer_key(name))
        row = conn.execute("SELECT report_json FROM issuers WHERE id = ?", (iid,)).fetchone() if iid else None
    if row:
        group["library_id"] = iid
        context = _report_context(json.loads(row["report_json"]))
    else:
        search = group["search_name"] or name
        year = datetime.now().year
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=4.0)) as client:
            r1, r2 = await asyncio.gather(
                brave_search(client, f"{search} Moody's S&P Fitch rating outlook", count=6),
                brave_search(client, f"{search} downgrade negative outlook deficit lawsuit {year}", count=5, freshness="py"),
            )
        context = "\n\n".join(x for x in [format_search_results(r1, "Web: ratings & outlooks"),
                                         format_search_results(r2, "Web (past 12 months): negative developments")] if x)
    prompt = (f"Today is {datetime.now().strftime('%B %d, %Y')}.\n\n{group['section']}\n\n"
              f"CUSIPs held from this issuer: {', '.join(group['cusips'])}\n\n{context or '(no evidence found)'}")
    try:
        out = await claude_json(SCAN_SYSTEM, prompt, max_tokens=6000)
    except Exception as e:
        out = {"issuer": name, "risk_score": None, "key_risk": f"Scan failed ({e}) — review manually", "flags": ["scan failed"]}
    return out


def _clamp(v, lo=1, hi=5):
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except Exception:
        return None


def portfolio_stats(rows: list, total_par: float) -> dict:
    def share(pred):
        return round(100 * sum(r["par"] for r in rows if pred(r)) / total_par, 1) if total_par else 0.0

    def by(field):
        acc = {}
        for r in rows:
            acc[r.get(field) or "Unknown"] = acc.get(r.get(field) or "Unknown", 0) + r["par"]
        return [{"name": k, "pct": round(100 * v / total_par, 1)} for k, v in sorted(acc.items(), key=lambda x: -x[1])]

    neg = lambda r: any(re.search(r"neg|watch|review", str((r["ratings"].get(a) or {}).get("outlook", "")), re.I)
                        for a in ("moodys", "sp", "fitch"))
    scored = [r for r in rows if r.get("risk_score")]
    wavg = (sum(r["risk_score"] * r["par"] for r in scored) / sum(r["par"] for r in scored)) if scored else None
    return {
        "total_par": total_par,
        "holdings": sum(len(r["cusips"]) for r in rows),
        "issuers": len(rows),
        "top_issuers": [{"name": r["issuer"], "pct": r["pct"]} for r in sorted(rows, key=lambda r: -r["par"])[:5]],
        "by_sector": by("sector"),
        "by_state": by("state"),
        "by_rating": by("bucket"),
        "pct_below_A": share(lambda r: r["notch"] is not None and r["notch"] >= 8),
        "pct_below_IG": share(lambda r: r["notch"] is not None and r["notch"] >= 11),
        "pct_unrated": share(lambda r: r["notch"] is None),
        "pct_negative_outlook": share(neg),
        "pct_high_risk": share(lambda r: (r.get("risk_score") or 0) >= 4),
        "pct_high_downgrade_risk": share(lambda r: (r.get("downgrade_risk") or 0) >= 4),
        "pct_high_spread_beta": share(lambda r: r.get("spread_beta") == "high"),
        "weighted_risk_score": round(wavg, 2) if wavg else None,
    }


async def run_portfolio(job: dict, holdings: list):
    try:
        cusips = [h["cusip"] for h in holdings]
        job.update(status="running", stage="Identifying holdings", done=0, total=len(cusips))
        await openfigi_batch(cusips)

        sem = asyncio.Semaphore(4)
        identities = {}

        async def ident(c):
            async with sem:
                identities[c] = await resolve_identity(c)
                job["done"] += 1
        await asyncio.gather(*[ident(c) for c in cusips])

        # Group holdings by issuer/obligor
        groups = {}
        par_by_cusip = {h["cusip"]: h["par"] for h in holdings}
        for c in cusips:
            idt = identities[c]
            if idt["data_source"] == "none":
                key = f"UNKNOWN-{c}"
            else:
                key = issuer_key(idt.get("official_name") or idt["issuer_name"])
            g = groups.setdefault(key, {"cusips": [], "par": 0.0, "issuer_name": idt["issuer_name"],
                                        "official_name": idt.get("official_name", ""), "search_name": idt.get("search_name", ""),
                                        "section": idt["section"], "unknown": idt["data_source"] == "none"})
            g["cusips"].append(c)
            g["par"] += par_by_cusip[c]

        known = [g for g in groups.values() if not g["unknown"]]
        job.update(stage="Scanning issuer credits", done=0, total=len(known))
        sem2 = asyncio.Semaphore(SCAN_CONCURRENCY)

        async def scan(g):
            async with sem2:
                g["scan"] = await scan_issuer(g)
                job["done"] += 1
        await asyncio.gather(*[scan(g) for g in known])

        total_par = sum(g["par"] for g in groups.values()) or 1.0
        rows = []
        for g in groups.values():
            sc = g.get("scan") or {}
            ratings = sc.get("ratings") or {}
            notches = [n for n in (rating_notch((ratings.get(a) or {}).get("rating")) for a in ("moodys", "sp", "fitch")) if n]
            lowest = max(notches) if notches else None
            rows.append({
                "issuer": sc.get("issuer") or g["official_name"] or g["issuer_name"] or f"Unidentified — {g['cusips'][0]}",
                "cusips": g["cusips"], "par": g["par"], "pct": round(100 * g["par"] / total_par, 1),
                "sector": sc.get("sector") or ("Unknown" if g["unknown"] else ""), "state": sc.get("state") or "",
                "ratings": {a: (ratings.get(a) or {}) for a in ("moodys", "sp", "fitch")},
                "notch": lowest, "bucket": rating_bucket(lowest) if not g["unknown"] else "Unidentified",
                "risk_score": _clamp(sc.get("risk_score")), "downgrade_risk": _clamp(sc.get("downgrade_risk")),
                "spread_beta": sc.get("spread_beta") if sc.get("spread_beta") in ("low", "medium", "high") else None,
                "key_risk": sc.get("key_risk") or ("Issuer could not be identified — verify CUSIP" if g["unknown"] else ""),
                "flags": sc.get("flags") or (["unidentified"] if g["unknown"] else []),
                "library_id": g.get("library_id"), "unknown": g["unknown"],
            })
        rows.sort(key=lambda r: (-(r["risk_score"] or 0), -r["par"]))
        stats = portfolio_stats([r for r in rows], total_par)

        job.update(stage="Building portfolio risk view", done=0, total=1)
        table = "\n".join(
            f"- {r['issuer']} | {r['pct']}% of par | {r['sector']} | {r['state']} | "
            f"Mdy {r['ratings']['moodys'].get('rating', 'NR')} {r['ratings']['moodys'].get('outlook', '')} / "
            f"S&P {r['ratings']['sp'].get('rating', 'NR')} {r['ratings']['sp'].get('outlook', '')} / "
            f"Fitch {r['ratings']['fitch'].get('rating', 'NR')} {r['ratings']['fitch'].get('outlook', '')} | "
            f"risk {r['risk_score']} | downgrade {r['downgrade_risk']} | spread beta {r['spread_beta']} | "
            f"{r['key_risk']} | flags: {', '.join(r['flags'])}" for r in rows)
        prompt = (f"Portfolio statistics (par-weighted):\n{json.dumps({k: v for k, v in stats.items() if k != 'total_par'})}\n\n"
                  f"Holdings by issuer (riskiest first):\n{table}")
        try:
            view = await claude_json(PORTFOLIO_SYSTEM, prompt, max_tokens=8000)
        except Exception as e:
            view = {"summary": f"Portfolio view could not be generated ({e}). Holding-level scan below is complete.",
                    "top_risks": [], "strengths": [], "conservativeness": {}}
        cons = view.get("conservativeness") or {}
        cons["rank"] = _clamp(cons.get("rank"))
        view["conservativeness"] = cons
        for r_ in view.get("top_risks") or []:
            r_["severity"] = _clamp(r_.get("severity"))
        job["done"] = 1
        job.update(status="done", stage="Done", result={"view": view, "stats": stats, "rows": rows,
                                                         "invalid": job.get("invalid", []),
                                                         "seconds": round(time.time() - job["started"])})
    except Exception as e:
        job.update(status="error", error=str(e))


class PortfolioRequest(BaseModel):
    holdings: list


@app.post("/api/portfolio/start")
async def portfolio_start(req: PortfolioRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")
    agg, invalid = {}, []
    for h in req.holdings[:300]:
        c = str((h or {}).get("cusip", "")).strip().upper()
        try:
            par = float(re.sub(r"[^0-9.\-]", "", str((h or {}).get("par", "0"))) or 0)
        except ValueError:
            par = 0
        if not cusip_valid(c) or par <= 0:
            invalid.append({"cusip": c, "par": (h or {}).get("par"), "reason": "invalid CUSIP" if not cusip_valid(c) else "no quantity"})
            continue
        agg[c] = agg.get(c, 0) + par
    if not agg:
        raise HTTPException(status_code=400, detail="No valid CUSIP / quantity rows found")
    if len(agg) > 150:
        raise HTTPException(status_code=400, detail="Please scan at most 150 CUSIPs at a time")
    # forget jobs older than 2 hours
    for jid in [j for j, v in _JOBS.items() if time.time() - v["started"] > 7200]:
        _JOBS.pop(jid, None)
    jid = f"{int(time.time() * 1000):x}"
    job = {"id": jid, "status": "queued", "stage": "Queued", "done": 0, "total": len(agg),
           "started": time.time(), "invalid": invalid}
    _JOBS[jid] = job
    asyncio.create_task(run_portfolio(job, [{"cusip": c, "par": p} for c, p in agg.items()]))
    return {"job_id": jid, "valid": len(agg), "invalid": invalid}


@app.get("/api/portfolio/{job_id}")
def portfolio_status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Scan not found (it may have expired — please rerun)")
    return {k: v for k, v in job.items() if k != "started"} | {"elapsed": round(time.time() - job["started"])}


# ── Recent muni rating actions (banner) ───────────────────────────────────────
# Built from news search: highlights, not a complete feed. Cached for 4 hours.

_ACTIONS_CACHE = {"at": 0.0, "items": []}
_ACTIONS_LOCK  = asyncio.Lock()
ACTIONS_TTL    = 4 * 3600
ACTIONS_DAYS   = 14

ACTION_QUERIES = [
    "Moody's upgrades municipal bonds",
    "Moody's downgrades municipal bonds",
    "S&P Global Ratings upgrades municipal bonds",
    "S&P Global Ratings downgrades municipal bonds",
    "Fitch upgrades municipal revenue bonds",
    "Fitch downgrades municipal bonds",
    "rating outlook revised positive negative municipal bonds",
    "KBRA assigns rating municipal bonds",
]


async def build_rating_actions() -> list:
    today = datetime.now()
    start = (today - timedelta(days=ACTIONS_DAYS)).strftime("%Y-%m-%d")
    window = f"{start}to{today.strftime('%Y-%m-%d')}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=4.0)) as client:
        batches = await asyncio.gather(*[brave_search(client, q, count=10, freshness=window) for q in ACTION_QUERIES],
                                       return_exceptions=True)
    seen, lines = set(), []
    for batch in batches:
        if isinstance(batch, Exception):
            continue
        for r in batch:
            if r["url"] in seen:
                continue
            seen.add(r["url"])
            text = " ".join([r["snippet"]] + list(r.get("extra") or []))[:350]
            lines.append(f"- [{(r.get('date') or '')[:10]}] {r['title']} | {text} | URL: {r['url']}")
    if not lines or not ANTHROPIC_API_KEY:
        return []

    prompt = (f"Today is {today.strftime('%B %d, %Y')}. From these search results, list US MUNICIPAL credit rating actions "
              f"from the last {ACTIONS_DAYS} days (state/local governments, school districts, public authorities, utilities, "
              "airports, toll roads, and nonprofit hospitals/universities/housing that issue municipal bonds). Exclude corporates, "
              "sovereigns, banks and structured finance. Only include an action actually described in a result; never invent. "
              "One entry per action (merge duplicates). Return ONLY JSON: {\"actions\": [{\"headline\": \"<= 90 chars, e.g. "
              "Moody's upgrades Chicago O'Hare airport revenue bonds to A1\", \"agency\": \"Moody's|S&P|Fitch|KBRA\", "
              "\"direction\": \"up|down|neutral\", \"date\": \"YYYY-MM-DD\", \"url\": \"source URL from the result\"}]}. "
              "direction: up = upgrade or outlook/watch to positive; down = downgrade or outlook/watch to negative; neutral = "
              "affirmation or new rating. Newest first, at most 25.\n\n" + "\n".join(lines[:80]))
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": NAME_MODEL, "max_tokens": 6000, "messages": [{"role": "user", "content": prompt}]},
            )
        if resp.status_code != 200:
            return []
        text = "".join(b.get("text", "") for b in resp.json().get("content", []) if b.get("type") == "text")
        m = re.search(r"\{[\s\S]*\}", text)
        actions = (json.loads(m.group()) if m else {}).get("actions", [])
    except Exception:
        return []

    cutoff = (today - timedelta(days=ACTIONS_DAYS + 1)).strftime("%Y-%m-%d")
    good = []
    for a in actions:
        if not isinstance(a, dict) or not a.get("headline") or not str(a.get("url", "")).startswith("http"):
            continue
        if a.get("date") and str(a["date"]) < cutoff:
            continue
        a["direction"] = a.get("direction") if a.get("direction") in ("up", "down", "neutral") else "neutral"
        good.append({k: str(a.get(k, "")) for k in ("headline", "agency", "direction", "date", "url")})
    good.sort(key=lambda a: a["date"], reverse=True)
    return good[:25]


@app.get("/api/rating-actions")
async def rating_actions(refresh: bool = False):
    async with _ACTIONS_LOCK:
        ttl = ACTIONS_TTL if _ACTIONS_CACHE["items"] else 30 * 60   # retry sooner if the last attempt came back empty
        if refresh or time.time() - _ACTIONS_CACHE["at"] > ttl:
            items = await build_rating_actions()
            if items or not _ACTIONS_CACHE["items"]:
                _ACTIONS_CACHE.update(at=time.time(), items=items)
            else:
                _ACTIONS_CACHE["at"] = time.time()   # keep the last good list rather than blanking the banner
    return {"as_of": datetime.fromtimestamp(_ACTIONS_CACHE["at"], timezone.utc).isoformat(timespec="seconds")
                     if _ACTIONS_CACHE["at"] else None,
            "days": ACTIONS_DAYS, "items": _ACTIONS_CACHE["items"]}


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
