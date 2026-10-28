import os
import json
import re
import asyncio
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

SYSTEM_PROMPT = """You are a senior municipal credit analyst at The Tribeca Group, a professional fixed income portfolio management team at Morgan Stanley. Your job is to produce rigorous, opinionated credit intelligence cards for muni bond CUSIPs. You have deep expertise in municipal credit structures, revenue pledge analysis, intercept mechanisms, sector benchmarks, and investability frameworks.

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
  "data_note": "Brief disclosure about data currency and sources used in this analysis — include whether live web data was retrieved and from what sources"
}

Be specific to the actual issuer. Use real figures where known. Do not give generic muni boilerplate. If data is uncertain, say so in the notes fields rather than omitting or fabricating. Distinguish between bond types within an issuer (e.g. LADWP Power vs Water; LAUSD GO vs COPs). When live web data is provided in the prompt, prioritize it over your training data for recent developments, ratings, and financial figures."""


# ── Web enrichment helpers ────────────────────────────────────────────────────

async def brave_search(client: httpx.AsyncClient, query: str, count: int = 5) -> list[dict]:
    """Run a Brave web search and return simplified result list."""
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
                "title": r.get("title", ""),
                "url":   r.get("url", ""),
                "snippet": r.get("description", ""),
            })
        return results
    except Exception:
        return []


async def emma_trade_data(client: httpx.AsyncClient, cusip: str) -> str:
    """Fetch recent trade activity for a CUSIP from MSRB EMMA."""
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
            date     = t.get("TradeDate", "")[:10]
            price    = t.get("Price", "N/A")
            yield_   = t.get("Yield", "N/A")
            par      = t.get("ParTraded", "N/A")
            side     = t.get("BuySellIndicator", "")
            lines.append(f"  {date} | {side} | Par: ${par:,} | Price: {price} | Yield: {yield_}%"
                         if isinstance(par, (int, float))
                         else f"  {date} | {side} | Par: {par} | Price: {price} | Yield: {yield_}%")
        return "\n".join(lines)
    except Exception:
        return ""


async def emma_security_info(client: httpx.AsyncClient, cusip: str) -> str:
    """Fetch security description from EMMA."""
    try:
        url = f"https://emma.msrb.org/api/Security/GetSecurityByCusip?cusip={cusip}"
        resp = await client.get(url, timeout=10.0, follow_redirects=True)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        sec = data.get("SecurityDetail", {})
        if not sec:
            return ""
        parts = []
        if sec.get("IssuerName"):
            parts.append(f"Issuer: {sec['IssuerName']}")
        if sec.get("Description"):
            parts.append(f"Description: {sec['Description']}")
        if sec.get("MaturityDate"):
            parts.append(f"Maturity: {sec['MaturityDate'][:10]}")
        if sec.get("CouponRate"):
            parts.append(f"Coupon: {sec['CouponRate']}%")
        if sec.get("TaxStatus"):
            parts.append(f"Tax status: {sec['TaxStatus']}")
        if sec.get("CallDate"):
            parts.append(f"First call: {sec['CallDate'][:10]} @ {sec.get('CallPrice', 'N/A')}")
        return "MSRB EMMA security details:\n  " + "\n  ".join(parts) if parts else ""
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


async def gather_live_data(cusip: str) -> str:
    """Run all enrichment fetches in parallel, hard cap 14s total."""
    async def _fetch_all():
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=4.0)) as client:
            issuer_query  = f'"{cusip}" municipal bond issuer credit rating 2024 2025'
            ratings_query = f'"{cusip}" rating action Moody\'s S&P Fitch muni 2024 2025'
            news_query    = f'"{cusip}" municipal bond news 2025'
            return await asyncio.gather(
                brave_search(client, issuer_query,  count=3),
                brave_search(client, ratings_query, count=3),
                brave_search(client, news_query,    count=3),
                emma_security_info(client, cusip),
                emma_trade_data(client, cusip),
                return_exceptions=True,
            )

    try:
        results = await asyncio.wait_for(_fetch_all(), timeout=14.0)
    except asyncio.TimeoutError:
        return ""

    issuer_results, ratings_results, news_results, emma_sec, emma_trades = [
        r if not isinstance(r, Exception) else ([] if i < 3 else "")
        for i, r in enumerate(results)
    ]

    sections = []

    if emma_sec:
        sections.append(emma_sec)

    if emma_trades:
        sections.append(emma_trades)

    issuer_block  = format_search_results(issuer_results,  "Recent web: issuer & credit")
    ratings_block = format_search_results(ratings_results, "Recent web: rating actions")
    news_block    = format_search_results(news_results,    "Recent web: news & developments")

    for block in [issuer_block, ratings_block, news_block]:
        if block:
            sections.append(block)

    if not sections:
        return ""

    return "\n\n## Live Data Retrieved at Request Time\n" + "\n\n".join(sections) + "\n"


# ── Request model ─────────────────────────────────────────────────────────────

class CreditIntelRequest(BaseModel):
    cusip: str
    pdf_text: Optional[str] = None
    model: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "MuniRadar Credit Intel API",
        "web_enrichment": bool(BRAVE_API_KEY),
    }


@app.post("/api/credit-intel")
async def credit_intel(req: CreditIntelRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    cusip = req.cusip.strip().upper()
    model = req.model or DEFAULT_MODEL

    # ── Parallel: gather live web data while we build the prompt ──
    live_data = await gather_live_data(cusip)

    pdf_section = ""
    if req.pdf_text:
        pdf_section = f"\n\n## Moody's Research Report (uploaded by analyst)\n{req.pdf_text[:12000]}\n"

    user_message = f"""Analyze CUSIP: {cusip}

Please generate a complete credit intelligence card for this municipal bond CUSIP. Use your knowledge of municipal credit markets, public filings, rating agency reports, and any relevant recent developments.{live_data}{pdf_section}

Key areas to address:
1. Identify the issuer and bond structure from this CUSIP (use EMMA data above if available)
2. Revenue sources and security pledge specifics
3. Key financial metrics vs sector benchmarks (days cash on hand, DSCR, debt/revenue, etc. as applicable to the sector)
4. Credit strengths and risks — be candid
5. Recent developments (use the live web data above — prioritize it over training data for anything dated 2024 or later)
6. All three ratings agency ratings and outlooks (use live rating data above if available)
7. Investability by maturity tier — be granular (e.g. short paper OK but avoid long end)
8. In data_note, mention whether live EMMA and web data was retrieved for this analysis"""

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
    raw = data["content"][0]["text"].strip()

    match = re.search(r'\{[\s\S]*\}', raw)
    if not match:
        raise HTTPException(status_code=500, detail="Could not parse model response as JSON")

    try:
        intel = json.loads(match.group())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error: {e}")

    return intel
