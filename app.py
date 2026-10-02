"""SARAP — акция талдау қосымшасы (Flask + yfinance)."""
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yfinance as yf
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="")

_CACHE = {}
CACHE_TTL = 600  # 10 минут


def cached(key, fn):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < CACHE_TTL:
        return hit[1]
    val = fn()
    _CACHE[key] = (now, val)
    return val


def num(x):
    try:
        if x is None:
            return None
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- history
def get_history(t):
    h = t.history(period="5y", interval="1d", auto_adjust=True)
    if h is None or h.empty:
        return [], []
    h = h.dropna(subset=["Close"])
    dates = [d.strftime("%Y-%m-%d") for d in h.index]
    closes = [round(float(c), 4) for c in h["Close"]]
    return dates, closes


def volatility(closes, days=252):
    arr = np.array(closes[-(days + 1):], dtype=float)
    if len(arr) < 30:
        return None, None
    r = np.diff(np.log(arr))
    sigma = float(np.std(r, ddof=1) * math.sqrt(252))
    mu = float(np.mean(r) * 252)
    return sigma, mu


def change(closes, n):
    if len(closes) > n and closes[-n - 1]:
        return closes[-1] / closes[-n - 1] - 1
    return None


# ---------------------------------------------------------------- news
def parse_news(raw):
    out = []
    for n in (raw or [])[:12]:
        c = n.get("content") if isinstance(n, dict) else None
        if c:  # жаңа yfinance форматы
            url = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
            out.append({
                "title": c.get("title"),
                "summary": c.get("summary") or c.get("description") or "",
                "source": (c.get("provider") or {}).get("displayName", ""),
                "date": (c.get("pubDate") or c.get("displayTime") or "")[:10],
                "url": url,
            })
        elif isinstance(n, dict):  # ескі формат
            ts = n.get("providerPublishTime")
            out.append({
                "title": n.get("title"),
                "summary": "",
                "source": n.get("publisher", ""),
                "date": time.strftime("%Y-%m-%d", time.gmtime(ts)) if ts else "",
                "url": n.get("link"),
            })
    return [x for x in out if x.get("title")]


# ---------------------------------------------------------------- peers
def peer_row(sym):
    try:
        i = yf.Ticker(sym).info or {}
        price = num(i.get("currentPrice") or i.get("regularMarketPrice"))
        tm = num(i.get("targetMeanPrice"))
        return {
            "symbol": sym,
            "name": i.get("shortName") or i.get("longName") or sym,
            "price": price,
            "marketCap": num(i.get("marketCap")),
            "revenueGrowth": num(i.get("revenueGrowth")),
            "profitMargin": num(i.get("profitMargins")),
            "ps": num(i.get("priceToSalesTrailing12Months")),
            "forwardPE": num(i.get("forwardPE")),
            "change1y": num(i.get("52WeekChange")),
            "upside": (tm / price - 1) if (tm and price) else None,
            "rec": i.get("recommendationKey"),
        }
    except Exception:
        return {"symbol": sym, "name": sym}


def get_peers(info, symbol, limit=6):
    key = info.get("industryKey")
    syms = []
    if key:
        try:
            tc = yf.Industry(key).top_companies
            if tc is not None and not tc.empty:
                syms = [s for s in tc.index.tolist() if s.upper() != symbol.upper()]
        except Exception:
            syms = []
    syms = syms[:limit]
    if not syms:
        return []
    with ThreadPoolExecutor(max_workers=6) as ex:
        return list(ex.map(peer_row, syms))


# ---------------------------------------------------------------- earnings date
def next_earnings(t):
    try:
        cal = t.calendar
        if isinstance(cal, dict):
            d = cal.get("Earnings Date")
            if d:
                d0 = d[0] if isinstance(d, (list, tuple)) else d
                return str(d0)[:10]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------- score
def score(m):
    """Қарапайым эвристикалық баға 0–100. Кепілдік емес."""
    parts = []

    def add(name, val, good, bad, weight=1.0):
        # good < bad болса (P/S, қарыз) — төмен мән жақсы деп саналады
        if val is None:
            return
        x = max(0.0, min(1.0, (val - bad) / (good - bad)))
        parts.append((name, x, weight))

    add("Түсім өсімі", m.get("revenueGrowth"), 0.30, -0.05, 1.5)
    add("Таза маржа", m.get("profitMargin"), 0.20, -0.20, 1.2)
    add("Жалпы маржа", m.get("grossMargin"), 0.70, 0.20, 0.8)
    add("Бағалау (P/S)", m.get("ps"), 2.0, 15.0, 1.0)  # төмен болса жақсы
    add("Қарыз/Ақша", m.get("debtToCash"), 0.3, 3.0, 0.8)  # төмен болса жақсы
    add("Талдаушы әлеуеті", m.get("upside"), 0.40, -0.20, 1.2)
    add("Моментум 6 ай", m.get("change6m"), 0.40, -0.40, 0.7)
    if not parts:
        return None, []
    total = sum(x * w for _, x, w in parts) / sum(w for _, _, w in parts)
    return round(total * 100), [{"name": n, "value": round(x * 100)} for n, x, _ in parts]


# ---------------------------------------------------------------- main
def analyze(symbol):
    t = yf.Ticker(symbol)
    info = t.info or {}
    dates, closes = get_history(t)
    if not closes and not info.get("regularMarketPrice"):
        raise ValueError("Тикер табылмады")

    price = num(info.get("currentPrice") or info.get("regularMarketPrice")) or (closes[-1] if closes else None)
    sigma, mu = volatility(closes)
    cash = num(info.get("totalCash"))
    debt = num(info.get("totalDebt"))
    tl, tmn, th = num(info.get("targetLowPrice")), num(info.get("targetMeanPrice")), num(info.get("targetHighPrice"))

    m = {
        "symbol": symbol.upper(),
        "name": info.get("longName") or info.get("shortName") or symbol.upper(),
        "exchange": info.get("exchange"),
        "currency": info.get("currency", "USD"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "website": info.get("website"),
        "employees": info.get("fullTimeEmployees"),
        "summary": info.get("longBusinessSummary"),
        "price": price,
        "prevClose": num(info.get("previousClose") or info.get("regularMarketPreviousClose")),
        "marketCap": num(info.get("marketCap")),
        "low52": num(info.get("fiftyTwoWeekLow")),
        "high52": num(info.get("fiftyTwoWeekHigh")),
        "beta": num(info.get("beta")),
        "revenue": num(info.get("totalRevenue")),
        "revenueGrowth": num(info.get("revenueGrowth")),
        "earningsGrowth": num(info.get("earningsGrowth")),
        "grossMargin": num(info.get("grossMargins")),
        "profitMargin": num(info.get("profitMargins")),
        "netIncome": num(info.get("netIncomeToCommon")),
        "freeCashflow": num(info.get("freeCashflow")),
        "cash": cash,
        "debt": debt,
        "debtToCash": (debt / cash) if (debt is not None and cash) else None,
        "trailingPE": num(info.get("trailingPE")),
        "forwardPE": num(info.get("forwardPE")),
        "ps": num(info.get("priceToSalesTrailing12Months")),
        "pb": num(info.get("priceToBook")),
        "eps": num(info.get("trailingEps")),
        "shortFloat": num(info.get("shortPercentOfFloat")),
        "insiders": num(info.get("heldPercentInsiders")),
        "institutions": num(info.get("heldPercentInstitutions")),
        "targetLow": tl,
        "targetMean": tmn,
        "targetHigh": th,
        "analysts": info.get("numberOfAnalystOpinions"),
        "recommendation": info.get("recommendationKey"),
        "upside": (tmn / price - 1) if (tmn and price) else None,
        "sigma": sigma,
        "mu": mu,
        "change1m": change(closes, 21),
        "change6m": change(closes, 126),
        "change1y": change(closes, 252),
        "nextEarnings": next_earnings(t),
        "dates": dates,
        "closes": closes,
    }
    m["score"], m["scoreParts"] = score(m)

    try:
        m["news"] = parse_news(t.news)
    except Exception:
        m["news"] = []
    try:
        m["peers"] = get_peers(info, symbol)
    except Exception:
        m["peers"] = []
    return m


@app.get("/api/analyze")
def api_analyze():
    sym = (request.args.get("t") or "").strip().upper()
    if not sym or len(sym) > 15:
        return jsonify({"error": "Тикер енгізіңіз"}), 400
    try:
        return jsonify(cached(sym, lambda: analyze(sym)))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"{sym}: деректер алынбады ({e})"}), 502


@app.get("/api/search")
def api_search():
    q = (request.args.get("q") or "").strip()
    if len(q) < 2:
        return jsonify([])
    try:
        res = yf.Search(q, max_results=6, news_count=0).quotes
        return jsonify([
            {"symbol": r.get("symbol"), "name": r.get("shortname") or r.get("longname"), "exch": r.get("exchDisp")}
            for r in res if r.get("quoteType") == "EQUITY"
        ])
    except Exception:
        return jsonify([])


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
