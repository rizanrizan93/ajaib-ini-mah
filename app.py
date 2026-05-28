import warnings
warnings.filterwarnings("ignore")

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, quote_plus
from io import BytesIO

# ==============================================================================
# SMART ALPHA SCREENER v17.0
# - Custom Universe + Proteksi 100 Emiten Scan
# - Fail-safe untuk saham illiquid / short-data
# - Fundamental + Growth + Value + Trader + SMC/ICT terintegrasi
# - Fokus strategi: Komprehensif / Trader / Growth / Value
# ==============================================================================

st.set_page_config(page_title="Smart Alpha Screener v17.0", layout="wide")

MAX_SCAN_UNIVERSE = 100
SCAN_MAX_WORKERS = 10


# ------------------------------------------------------------------------------
# 1. SESSION STATE INITIALIZATION
# ------------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "app_mode": "Light Scan Multi-Saham",
        "active_universe": ["BBRI.JK", "BBCA.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK", "NCKL.JK", "SSIA.JK"],
        "target_ticker": "BBRI",
        "cached_scan_results": None,
        "selected_status_filter": "Semua",
        "min_final_score": 0.0,
        "universe_search": "",
        "select_all_filtered": False,
        "auto_deep_dive": False,
        "strategy_focus": "Komprehensif",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ------------------------------------------------------------------------------
# 2. BASE UNIVERSE DEFINITION
# ------------------------------------------------------------------------------
def unique_keep_order(items):
    out, seen = [], set()
    for x in items:
        x = str(x).strip().upper()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


ALL_IHSG_RAW = unique_keep_order([
    "BBRI", "BBCA", "BMRI", "BBNI", "BRIS", "BBTN", "BDMN", "BNGA", "NISP", "MEGA",
    "PNBN", "BTPS", "AGRO", "ARTO", "BFIN", "DNAR", "BCIC", "BVIC", "BBKP", "BABP",
    "TLKM", "ISAT", "EXCL", "MTEL", "TBIG", "LINK", "FREN", "GOTO", "BUKA", "WIFI",
    "ASII", "AUTO", "IMAS", "BIRD", "SMSM", "GJTL", "TURI", "MAPA", "MAPI", "ACES",
    "AMRT", "MIDI", "LPPF", "RALS", "ERAA", "MTSM", "HERO", "LION", "WSBP", "WSKT",
    "PTPP", "WIKA", "WEGE", "SMRA", "PWON", "CTRA", "BSDE", "DMAS", "KIJA", "DILD",
    "LPKR", "ASRI", "TOTL", "NICK", "MKPI", "KPIG", "GMTD", "BAPA", "PANI", "MLBI", "YULE",
    "UNTR", "ADRO", "ITMG", "PTBA", "ANTM", "INCO", "NCKL", "MBMA", "MDKA", "TINS",
    "BYAN", "GEMS", "DOID", "HRUM", "AADI", "ENRG", "MEDC", "ELSA", "RAJA", "RATU",
    "PGAS", "AKRA", "RUIS", "SSIA", "INDY", "CUAN", "BIPI", "DSSA", "KOPI", "BSSR",
    "SMGR", "INTP", "SCCO", "ARNA", "TOTO", "AMFG", "WTON", "CPIN", "JPFA", "SIDO",
    "ICBP", "INDF", "UNVR", "MYOR", "ULTJ", "CLEO", "HOKI", "CAMP", "TSPC", "KLBF",
    "KAEF", "HEAL", "MIKA", "SILO", "PRDA", "PEHA", "SRAJ", "INAF", "KINO", "ADES",
    "TBLA", "LSIP", "SMAR", "AISA", "DSNG", "TAPG", "SSMS", "SGRO", "SIMP", "BWPT",
    "CPRO", "MARK", "SCMA", "MNCN", "FILM", "CNMA", "ELPI", "BISI", "HRTA", "PTMP"
])
ALL_IHSG = [f"{t}.JK" for t in ALL_IHSG_RAW]


# ------------------------------------------------------------------------------
# 3. ROBUST HELPERS & UTILITIES
# ------------------------------------------------------------------------------
def safe_num(x, default=np.nan):
    try:
        if x is None or pd.isna(x):
            return default
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        if isinstance(x, str):
            cleaned = x.replace(",", "").replace("%", "").strip()
            return float(cleaned)
        return float(x)
    except Exception:
        return default


def fmt_num(x, decimals=2, suffix=""):
    if x is None or pd.isna(x):
        return "N/A"
    try:
        return f"{float(x):,.{decimals}f}{suffix}"
    except Exception:
        return "N/A"


def pct(x, decimals=1):
    if x is None or pd.isna(x):
        return "N/A"
    try:
        return f"{float(x):.{decimals}f}%"
    except Exception:
        return "N/A"


def strip_suffix(ticker: str) -> str:
    return re.sub(r"\.JK$", "", str(ticker), flags=re.IGNORECASE)


def ensure_yf_ticker(ticker: str) -> str:
    ticker = strip_suffix(ticker).upper().strip()
    return f"{ticker}.JK"


def normalize_label(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def unique_like_series(series: pd.Series):
    if series is None or series.empty:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce").dropna()


def safe_getattr(obj, names, default=None):
    for name in names:
        try:
            val = getattr(obj, name)
            if val is not None:
                return val
        except Exception:
            continue
    return default




# ------------------------------------------------------------------------------
# 3A. OFFICIAL COMPANY WEBSITE VALIDATION (BEST-EFFORT)
# ------------------------------------------------------------------------------
REQUEST_TIMEOUT = 12
MAX_OFFICIAL_PAGES = 4
MAX_OFFICIAL_DOCS = 3

OFFICIAL_KEYWORDS = {
    "annual_report": [
        "annual report", "laporan tahunan", "annualreport", "annual-report",
        "investor relations", "ir", "reports", "report", "annual"
    ],
    "financial_statement": [
        "financial statement", "financial statements", "laporan keuangan",
        "financial report", "interim report", "quarterly report", "unaudited"
    ],
    "revenue": [
        "total revenue", "revenue", "pendapatan", "net sales", "sales"
    ],
    "net_income": [
        "net income", "laba bersih", "profit for the year",
        "income for the year", "profit attributable"
    ],
    "gross_profit": [
        "gross profit", "laba kotor"
    ],
    "operating_income": [
        "operating income", "operating profit", "ebit", "income from operations"
    ],
    "equity": [
        "total equity", "equity", "ekuitas", "shareholders equity", "stockholders equity"
    ],
    "assets": [
        "total assets", "assets", "aset"
    ],
    "current_assets": [
        "current assets", "aset lancar"
    ],
    "current_liabilities": [
        "current liabilities", "liabilitas lancar"
    ],
    "debt": [
        "total debt", "debt", "borrowings", "utang berbunga", "interest-bearing debt"
    ],
    "ocf": [
        "operating cash flow", "cash flows from operating activities",
        "arus kas operasi", "cash flow from operating activities"
    ],
    "capex": [
        "capital expenditure", "capital expenditures", "capex",
        "purchases of property plant and equipment",
        "purchase of property plant and equipment",
        "belanja modal"
    ],
}

def normalize_url(url: str) -> str:
    url = str(url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url.lstrip("/")
    return url.rstrip("/")

def get_domain(url: str) -> str:
    try:
        parsed = urlparse(normalize_url(url))
        return parsed.netloc.lower().replace("www.", "")
    except Exception:
        return ""


def same_domain(url_a: str, url_b: str) -> bool:
    da = get_domain(url_a)
    db = get_domain(url_b)
    if not da or not db:
        return False
    return da == db or da.endswith("." + db) or db.endswith("." + da)

BLOCKED_SEARCH_DOMAINS = {
    "yahoo.com", "finance.yahoo.com", "stockbit.com", "id.investing.com", "investing.com",
    "tradingview.com", "marketbeat.com", "companiesmarketcap.com", "marketscreener.com",
    "wsj.com", "reuters.com", "bloomberg.com", "morningstar.com", "simplywall.st",
    "google.com", "bing.com", "duckduckgo.com", "facebook.com", "instagram.com",
    "x.com", "twitter.com", "linkedin.com", "youtube.com", "wikipedia.org",
    "idx.co.id", "idxchannel.com"
}

COMPANY_HINT_WORDS = (
    "investor relations", "investor relation", "annual report", "financial report",
    "financial statements", "annual report pdf", "laporan tahunan", "laporan keuangan",
    "company profile", "corporate", "official website"
)

def normalize_company_name(name: str) -> str:
    name = str(name or "").lower().strip()
    name = re.sub(r"\b(pt|tbk|persero|bk|inc|corp|corporation|ltd|limited|holdings?|group)\b", " ", name)
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()

def _extract_search_links(html: str, engine: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []

    if engine == "ddg":
        selectors = ["a.result__a", "a[data-testid='result-title-a']"]
    else:
        selectors = ["li.b_algo h2 a", "a[href]"]

    for sel in selectors:
        for a in soup.select(sel):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            text = " ".join(a.get_text(" ", strip=True).split())
            if not text:
                text = ""
            # DDG redirect unwrap
            if "duckduckgo.com/l/" in href or "duckduckgo.com/?uddg=" in href:
                parsed = urlparse(href)
                q = parsed.query
                m = re.search(r"(?:^|&)uddg=([^&]+)", q)
                if m:
                    try:
                        href = requests.utils.unquote(m.group(1))
                    except Exception:
                        pass
            links.append(href)
        if links:
            break

    clean = []
    seen = set()
    for u in links:
        if not u or u in seen:
            continue
        seen.add(u)
        clean.append(u)
    return clean

def _is_blocked_domain(url: str) -> bool:
    dom = get_domain(url)
    return any(dom == bd or dom.endswith("." + bd) for bd in BLOCKED_SEARCH_DOMAINS)

def _score_company_url(url: str, query_tokens: list[str], company_name: str = "") -> float:
    try:
        dom = get_domain(url)
        path = urlparse(url).path.lower()
        score = 0.0
        if not dom:
            return -999.0
        if _is_blocked_domain(url):
            return -100.0

        if dom.endswith(".co.id") or dom.endswith(".id"):
            score += 2.5
        if any(k in path for k in ["investor", "ir", "annual", "report", "financial", "corporate", "relations"]):
            score += 3.0
        if any(k in dom for k in ["investor", "ir", "corp", "company", "official"]):
            score += 2.0

        slug = normalize_company_name(company_name)
        if slug:
            slug_tokens = slug.split()
            matched = sum(1 for tok in slug_tokens if tok in dom or tok in path)
            score += min(4.0, matched * 1.2)

        for tok in query_tokens:
            tok = tok.lower().strip()
            if tok and tok in dom:
                score += 0.8
            if tok and tok in path:
                score += 0.5

        if "www." in dom:
            score += 0.1
        return score
    except Exception:
        return -999.0

@st.cache_data(ttl=86400, show_spinner=False)
def resolve_company_website(ticker: str, company_name: str = "", website_hint: str = "") -> dict:
    """
    Resolve official website from ticker/company name.
    Priority:
    1) yfinance website hint
    2) direct search query to DuckDuckGo/Bing
    3) infer from official-looking result
    """
    ticker = strip_suffix(ticker).upper().strip()
    website_hint = normalize_url(website_hint)

    candidates = []
    if website_hint:
        candidates.append(("yfinance", website_hint, 10.0))

    search_queries = []
    if company_name:
        search_queries.append(f"{company_name} official website investor relations")
        search_queries.append(f"{company_name} annual report investor relations")
    if ticker:
        search_queries.append(f"{ticker} official website investor relations Indonesia")
        search_queries.append(f"{ticker} annual report investor relations")

    query_tokens = [strip_suffix(ticker).lower()]
    query_tokens.extend(normalize_company_name(company_name).split())

    for q in search_queries[:4]:
        q = q.strip()
        if not q:
            continue

        for engine, search_url in (
            ("ddg", f"https://html.duckduckgo.com/html/?q={quote_plus(q)}"),
            ("bing", f"https://www.bing.com/search?q={quote_plus(q)}"),
        ):
            try:
                resp = requests.get(search_url, timeout=REQUEST_TIMEOUT, headers=_session_headers())
                if not resp.ok:
                    continue
                links = _extract_search_links(resp.text or "", engine=engine)
                for link in links[:10]:
                    if not link:
                        continue
                    # unwrap relative links if any
                    link = urljoin(search_url, link)
                    if not link.startswith(("http://", "https://")):
                        continue
                    if _is_blocked_domain(link):
                        continue
                    score = _score_company_url(link, query_tokens=query_tokens, company_name=company_name)
                    if score > 0:
                        candidates.append(("search", normalize_url(link), score))
            except Exception:
                continue

    # Deduplicate by URL while preserving best score.
    best_by_url = {}
    for source, url, score in candidates:
        if not url:
            continue
        cur = best_by_url.get(url)
        if cur is None or score > cur[2]:
            best_by_url[url] = (source, url, score)

    ranked = sorted(best_by_url.values(), key=lambda x: x[2], reverse=True)
    chosen = ranked[0] if ranked else ("", "", 0.0)

    return {
        "ticker": ticker,
        "company_name": company_name,
        "resolved_website": chosen[1],
        "source": chosen[0] or "none",
        "score": float(chosen[2]) if chosen else 0.0,
        "candidates": [u for _, u, _ in ranked[:8]],
    }

def _session_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )
    }

def fetch_html(session: requests.Session, url: str) -> tuple[str, str]:
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT, headers=_session_headers(), allow_redirects=True)
        if not resp.ok:
            return "", ""
        resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"
        return resp.text or "", resp.url or url
    except Exception:
        return "", ""

def parse_pdf_text_from_bytes(blob: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(blob))
        out = []
        for i, page in enumerate(reader.pages[:60]):
            try:
                txt = page.extract_text() or ""
                if txt:
                    out.append(txt)
            except Exception:
                continue
        return "\n".join(out)
    except Exception:
        return ""

def html_to_text(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text("\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
    except Exception:
        return html or ""

def download_and_extract_text(session: requests.Session, url: str) -> tuple[str, str]:
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT, headers=_session_headers(), allow_redirects=True)
        if not resp.ok:
            return "", ""
        final_url = resp.url or url
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "pdf" in ctype or final_url.lower().endswith(".pdf"):
            return parse_pdf_text_from_bytes(resp.content), final_url
        html = resp.text or ""
        text = html_to_text(html)

        # Append HTML tables if present; this helps statements published as table pages.
        try:
            tables = pd.read_html(html)
            for tbl in tables[:8]:
                try:
                    text += "\n" + tbl.to_string(index=False)
                except Exception:
                    continue
        except Exception:
            pass

        return text, final_url
    except Exception:
        return "", ""

def normalize_number_token(token: str) -> float:
    try:
        s = str(token).strip()
        if not s:
            return np.nan
        neg = False
        if s.startswith("(") and s.endswith(")"):
            neg = True
            s = s[1:-1]
        s = s.replace("Rp", "").replace("IDR", "").replace("USD", "").replace("$", "")
        s = s.replace(" ", "")
        s = re.sub(r"[^0-9,.\-]", "", s)
        if not s or s in {"-", ".", ","}:
            return np.nan

        # Handle mixed separators
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                # comma decimal, dot thousand
                s = s.replace(".", "").replace(",", ".")
            else:
                # dot decimal, comma thousand
                s = s.replace(",", "")
        elif s.count(",") > 1 and "." not in s:
            s = s.replace(",", "")
        elif s.count(".") > 1 and "," not in s:
            s = s.replace(".", "")
        else:
            # Single separator: if last part length suggests thousand separator, strip it.
            if "," in s:
                parts = s.split(",")
                if len(parts[-1]) == 3:
                    s = s.replace(",", "")
                else:
                    s = s.replace(",", ".")
            if "." in s:
                parts = s.split(".")
                if len(parts[-1]) == 3 and len(parts) > 2:
                    s = s.replace(".", "")
        val = float(s)
        if neg:
            val *= -1.0
        return float(val)
    except Exception:
        return np.nan

def extract_numeric_candidates(text: str) -> list[float]:
    if not text:
        return []
    tokens = re.findall(r"\(?-?\d[\d,.\s]*\)?", str(text))
    vals = []
    for tok in tokens:
        v = normalize_number_token(tok)
        if pd.notna(v):
            vals.append(float(v))
    return vals

def _drop_year_like(values: list[float]) -> list[float]:
    out = []
    for v in values:
        if 1900 <= abs(v) <= 2100 and float(v).is_integer():
            continue
        out.append(v)
    return out

def extract_metric_from_text(text: str, keywords: list[str]) -> float:
    if not text:
        return np.nan

    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in str(text).splitlines()]
    candidates = []

    for line in lines:
        norm_line = normalize_label(line)
        if not any(normalize_label(k) in norm_line for k in keywords):
            continue

        vals = _drop_year_like(extract_numeric_candidates(line))
        if not vals:
            continue

        # Prefer the first non-year number on the line. For table-like strings,
        # this is often the most relevant value adjacent to the row label.
        score = 0
        if any(k in norm_line for k in ["2024", "2025", "fy24", "fy25"]):
            score += 1
        if any(k in norm_line for k in ["consolidated", "audited", "unaudited"]):
            score += 1
        candidates.append((score, vals[0]))

    if not candidates:
        return np.nan

    # prefer cleaner lines, then larger magnitude as tie-breaker
    candidates.sort(key=lambda x: (x[0], abs(x[1])), reverse=True)
    return safe_num(candidates[0][1], np.nan)

def discover_official_sources(website: str) -> list[str]:
    website = normalize_url(website)
    if not website:
        return []

    session = requests.Session()
    visited = set()
    discovered = []

    seed_pages = [
        website,
        urljoin(website + "/", "investor-relations"),
        urljoin(website + "/", "investor"),
        urljoin(website + "/", "ir"),
        urljoin(website + "/", "reports"),
        urljoin(website + "/", "annual-report"),
        urljoin(website + "/", "financial-reports"),
        urljoin(website + "/", "financial-report"),
        urljoin(website + "/", "laporan-keuangan"),
        urljoin(website + "/", "annual-reports"),
    ]

    path_keywords = ("annual", "report", "laporan", "financial", "investor", "ir", "pdf", "results", "statements")
    direct_keywords = ("annual report", "laporan tahunan", "financial statements", "laporan keuangan", "interim", "quarterly")

    for page_url in seed_pages:
        if page_url in visited:
            continue
        visited.add(page_url)

        html, final_url = fetch_html(session, page_url)
        if not html:
            continue

        base_url = final_url or page_url
        soup = BeautifulSoup(html, "html.parser")
        anchors = soup.find_all("a", href=True)

        for a in anchors:
            href = a.get("href", "").strip()
            text = " ".join(a.get_text(" ", strip=True).split()).lower()
            full = urljoin(base_url + "/", href)
            if not same_domain(full, website):
                continue

            joined = f"{text} {href}".lower()
            if any(k in joined for k in direct_keywords + path_keywords) or full.lower().endswith(".pdf"):
                discovered.append(full)

        # Stop early if enough candidate documents found.
        if len(discovered) >= 12:
            break

    # Add common document paths if HTML crawl didn't reveal enough.
    fallback_docs = [
        "annual-report.pdf", "annualreport.pdf", "report.pdf",
        "investor-relations/annual-report.pdf",
        "investor/annual-report.pdf",
        "uploads/annual-report.pdf",
        "docs/annual-report.pdf",
        "financial-statements.pdf",
        "laporan-keuangan.pdf",
    ]
    for path in fallback_docs:
        discovered.append(urljoin(website + "/", path))

    # Deduplicate, same-domain only, preserve order
    clean = []
    seen = set()
    for u in discovered:
        u = u.strip()
        if not u or u in seen or not same_domain(u, website):
            continue
        seen.add(u)
        clean.append(u)
    return clean[:MAX_OFFICIAL_PAGES + MAX_OFFICIAL_DOCS]

def _select_best_metric(value_current, value_official, metric: str, sector_bucket_name: str = "general"):
    if is_plausible_metric_value(value_official, metric, sector_bucket_name):
        return safe_num(value_official), "official_web"
    if is_plausible_metric_value(value_current, metric, sector_bucket_name):
        return safe_num(value_current), "yfinance"
    return np.nan, "N/A"

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_official_company_fundamentals(website: str, ticker: str) -> dict:
    """
    Best-effort official company website validator:
    - crawls the company website / investor relations pages
    - finds annual report / financial statement links
    - extracts text from HTML or PDF
    - tries to recover core fundamentals from official sources
    """
    website = normalize_url(website)
    if not website:
        return {
            "website": "",
            "source_urls": [],
            "source_type": "no_website",
            "confidence": 0.0,
            "metrics": {},
            "raw_text": "",
        }

    source_urls = discover_official_sources(website)
    if not source_urls:
        return {
            "website": website,
            "source_urls": [],
            "source_type": "no_report_found",
            "confidence": 0.0,
            "metrics": {},
            "raw_text": "",
        }

    session = requests.Session()
    text_chunks = []
    used_urls = []

    for url in source_urls[:MAX_OFFICIAL_PAGES + MAX_OFFICIAL_DOCS]:
        txt, final_url = download_and_extract_text(session, url)
        if txt and len(txt.strip()) >= 300:
            text_chunks.append(txt)
            used_urls.append(final_url or url)

    raw_text = "\n\n".join(text_chunks)
    if not raw_text.strip():
        return {
            "website": website,
            "source_urls": used_urls,
            "source_type": "empty_after_download",
            "confidence": 0.0,
            "metrics": {},
            "raw_text": "",
        }

    metrics = {
        "rev_ttm": extract_metric_from_text(raw_text, OFFICIAL_KEYWORDS["revenue"]),
        "ni_ttm": extract_metric_from_text(raw_text, OFFICIAL_KEYWORDS["net_income"]),
        "gross_profit_ttm": extract_metric_from_text(raw_text, OFFICIAL_KEYWORDS["gross_profit"]),
        "ebit_ttm": extract_metric_from_text(raw_text, OFFICIAL_KEYWORDS["operating_income"]),
        "equity": extract_metric_from_text(raw_text, OFFICIAL_KEYWORDS["equity"]),
        "assets": extract_metric_from_text(raw_text, OFFICIAL_KEYWORDS["assets"]),
        "current_assets": extract_metric_from_text(raw_text, OFFICIAL_KEYWORDS["current_assets"]),
        "current_liabilities": extract_metric_from_text(raw_text, OFFICIAL_KEYWORDS["current_liabilities"]),
        "debt": extract_metric_from_text(raw_text, OFFICIAL_KEYWORDS["debt"]),
        "ocf_ttm": extract_metric_from_text(raw_text, OFFICIAL_KEYWORDS["ocf"]),
        "capex_ttm": extract_metric_from_text(raw_text, OFFICIAL_KEYWORDS["capex"]),
    }

    found = sum(1 for v in metrics.values() if pd.notna(v))
    confidence = min(1.0, found / 8.0)

    source_type = "company_website"
    if any(u.lower().endswith(".pdf") for u in used_urls):
        source_type = "annual_report_pdf"
    elif used_urls:
        source_type = "company_html"

    return {
        "website": website,
        "source_urls": used_urls[:8],
        "source_type": source_type,
        "confidence": confidence,
        "metrics": metrics,
        "raw_text": raw_text[:5000],
    }


def extract_metric(df: pd.DataFrame, keywords: List[str]) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    for kw in keywords:
        norm_kw = normalize_label(kw)
        for idx in df.index:
            if normalize_label(idx) == norm_kw:
                s = unique_like_series(df.loc[idx])
                if not s.empty:
                    return s.sort_index()

    for kw in keywords:
        norm_kw = normalize_label(kw)
        for idx in df.index:
            if norm_kw in normalize_label(idx):
                s = unique_like_series(df.loc[idx])
                if not s.empty:
                    return s.sort_index()

    return pd.Series(dtype="float64")


def latest_value(series: pd.Series, default=np.nan):
    if series is None or series.empty:
        return default
    try:
        return safe_num(series.iloc[-1], default)
    except Exception:
        return default


def get_ttm(q_series: pd.Series, a_series: pd.Series):
    if q_series is not None and not q_series.empty:
        if len(q_series) >= 4:
            return safe_num(q_series.iloc[-4:].sum())
        return safe_num(q_series.sum() * (4 / len(q_series)))
    if a_series is not None and not a_series.empty:
        return safe_num(a_series.iloc[-1])
    return np.nan


def yoy_growth(current: float, previous: float) -> float:
    current = safe_num(current)
    previous = safe_num(previous)
    if np.isnan(current) or np.isnan(previous) or previous == 0:
        return np.nan
    return ((current - previous) / abs(previous)) * 100.0


def growth_from_series(q_series: pd.Series, a_series: pd.Series = None) -> float:
    """
    Hitung growth YoY secara lebih fleksibel:
    - prioritas data kuartalan (TTM vs TTM tahun lalu) bila tersedia
    - fallback ke data annual bila kuartalan tidak cukup
    """
    q = pd.to_numeric(q_series, errors="coerce").dropna() if q_series is not None else pd.Series(dtype="float64")
    if len(q) >= 8:
        latest_4 = q.iloc[-4:].sum()
        prev_4 = q.iloc[-8:-4].sum()
        return yoy_growth(latest_4, prev_4)

    a = pd.to_numeric(a_series, errors="coerce").dropna() if a_series is not None else pd.Series(dtype="float64")
    if len(a) >= 2:
        return yoy_growth(a.iloc[-1], a.iloc[-2])

    return np.nan


def growth_from_quarter_series(s: pd.Series) -> float:
    return growth_from_series(s, None)


def normalize_percent_like(x):
    """Jika yfinance mengembalikan rasio desimal (0.17), ubah ke 17.0."""
    x = safe_num(x)
    if np.isnan(x):
        return np.nan
    if abs(x) <= 1.5:
        return x * 100.0
    return x


def normalize_debt_to_equity(info_ratio, calc_ratio, sector_bucket_name: str = "general"):
    """
    Pilih DER yang paling masuk akal dari sumber info dan kalkulasi manual.
    Untuk non-financial, prefer angka yang <= 5x.
    """
    candidates = []

    for ratio in (info_ratio, calc_ratio):
        ratio = safe_num(ratio)
        if np.isnan(ratio) or ratio <= 0:
            continue
        # Banyak source mengekspresikan DER dalam persen (mis. 29 = 0.29x)
        if ratio > 20 and ratio <= 5000:
            ratio = ratio / 100.0
        candidates.append(ratio)

    if not candidates:
        return np.nan

    sector_bucket_name = str(sector_bucket_name).lower().strip()

    if sector_bucket_name == "financial":
        plausible = [x for x in candidates if x > 0]
        return float(min(plausible)) if plausible else np.nan

    plausible = [x for x in candidates if x <= 5]
    if plausible:
        return float(min(plausible))

    return float(min(candidates))


def investment_status(score: float) -> str:
    if score >= 80:
        return "LAYAK BELI"
    if score >= 65:
        return "WATCHLIST"
    if score >= 50:
        return "NETRAL"
    return "HINDARI"


def status_reason(score: float, fund: dict, tech: pd.DataFrame, subscores: Dict[str, float] = None) -> str:
    last = tech.iloc[-1] if tech is not None and not tech.empty else None
    reasons = []
    if score >= 80:
        reasons.append("kualitas, growth, dan momentum selaras")
    elif score >= 65:
        reasons.append("setup menarik namun perlu konfirmasi")
    elif score >= 50:
        reasons.append("belum cukup kuat untuk entry agresif")
    else:
        reasons.append("probabilitas rendah / risiko tinggi")

    if subscores:
        top = sorted(subscores.items(), key=lambda x: x[1], reverse=True)[:3]
        reasons.append("top score: " + ", ".join([f"{k} {v:.0f}" for k, v in top]))

    if fund:
        roe = safe_num(fund.get("roe"))
        der = safe_num(fund.get("der"))
        revg = safe_num(fund.get("rev_growth_yoy"))
        if not np.isnan(roe):
            reasons.append(f"ROE {roe:.1f}%")
        if not np.isnan(der):
            reasons.append(f"DER {der:.2f}x")
        if not np.isnan(revg):
            reasons.append(f"Revenue Growth {revg:.1f}%")

    if last is not None:
        if bool(last.get("Bullish_ChoCH")):
            reasons.append("ChoCH bullish")
        if bool(last.get("Liquidity_Sweep")):
            reasons.append("liquidity sweep valid")
        if bool(last.get("Bullish_OB")):
            reasons.append("order block terdeteksi")

    return " | ".join(reasons[:5])


def sector_bucket(fund: dict) -> str:
    sector = str(fund.get("sector", "")).lower()
    industry = str(fund.get("industry", "")).lower()
    text = f"{sector} {industry}"
    if any(k in text for k in ["bank", "insurance", "financial", "financing", "broker"]):
        return "financial"
    if any(k in text for k in ["coal", "oil", "gas", "commodity", "metal", "mining", "energy"]):
        return "cyclical"
    if any(k in text for k in ["consumer", "beverage", "food", "household", "pharma", "retail"]):
        return "defensive"
    if any(k in text for k in ["property", "real estate", "construction"]):
        return "property"
    return "general"


# ------------------------------------------------------------------------------
# 4. DATA ENGINE (FUNDAMENTAL & PRICE FETCH)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_price(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=False, actions=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=str.title)
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        if len(cols) < 5:
            return pd.DataFrame()
        return df[cols].dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()




def is_plausible_metric_value(value, metric: str, sector_bucket_name: str = "general") -> bool:
    """
    Filter kasar agar angka ekstrem tidak dipakai sebagai input utama.
    Tujuannya bukan akurasi akademik, tetapi mencegah rasio absurd masuk ke dashboard.
    """
    try:
        v = safe_num(value)
        if np.isnan(v) or np.isinf(v):
            return False

        metric = str(metric).lower().strip()
        sector_bucket_name = str(sector_bucket_name).lower().strip()

        if metric in {"pb", "pbv"}:
            if v <= 0:
                return False
            return v <= (30 if sector_bucket_name == "financial" else 20)

        if metric in {"pe"}:
            return 0 < v <= 100

        if metric in {"der"}:
            return 0 <= v <= (30 if sector_bucket_name == "financial" else 10)

        if metric in {"roe", "roa", "npm", "margin", "growth"}:
            return -500 <= v <= 500

        if metric in {"cr", "currentratio"}:
            return 0 <= v <= 50

        return True
    except Exception:
        return False


def choose_first_plausible(candidates, metric: str, sector_bucket_name: str = "general"):
    """
    candidates: list of tuples (label, value)
    Memilih kandidat pertama yang masuk akal.
    """
    for label, value in candidates:
        if is_plausible_metric_value(value, metric, sector_bucket_name):
            return safe_num(value), str(label)
    return np.nan, "N/A"

@st.cache_data(ttl=14400, show_spinner=False)
def fetch_fundamentals(ticker: str, use_official_web_validation: bool = True, website_override: str = "", auto_discover_website: bool = True) -> dict:
    try:
        tk = yf.Ticker(ticker)
        info = {}

        try:
            raw_info = tk.info or {}
            if hasattr(raw_info, "items"):
                info.update(dict(raw_info))
        except Exception:
            pass

        try:
            raw_fast = getattr(tk, "fast_info", {}) or {}
            if hasattr(raw_fast, "items"):
                info.update(dict(raw_fast))
        except Exception:
            pass

        company_name = str(info.get("longName") or info.get("shortName") or ticker).strip()
        website_hint = normalize_url(website_override or info.get("website", ""))
        resolved_web = ""
        web_resolve_bundle = {"resolved_website": "", "source": "none", "score": 0.0, "candidates": []}

        if auto_discover_website:
            try:
                web_resolve_bundle = resolve_company_website(ticker, company_name=company_name, website_hint=website_hint)
                resolved_web = web_resolve_bundle.get("resolved_website", "") or ""
            except Exception:
                resolved_web = website_hint
        else:
            resolved_web = website_hint

        if not resolved_web:
            resolved_web = website_hint

        official_bundle = {}
        if use_official_web_validation:
            try:
                official_bundle = fetch_official_company_fundamentals(resolved_web or "", ticker)
            except Exception:
                official_bundle = {}
        official_metrics = (official_bundle or {}).get("metrics", {}) if isinstance(official_bundle, dict) else {}

        # Quarterly frames
        q_inc = safe_getattr(tk, ["quarterly_income_stmt", "quarterly_financials"], pd.DataFrame())
        q_bal = safe_getattr(tk, ["quarterly_balance_sheet"], pd.DataFrame())
        q_cf = safe_getattr(tk, ["quarterly_cashflow", "quarterly_cash_flow"], pd.DataFrame())

        # Annual / fallback frames
        a_inc = safe_getattr(tk, ["income_stmt", "financials", "annual_income_stmt", "annual_financials"], pd.DataFrame())
        a_bal = safe_getattr(tk, ["balance_sheet", "annual_balance_sheet"], pd.DataFrame())
        a_cf = safe_getattr(tk, ["cashflow", "cash_flow", "annual_cashflow", "annual_cash_flow"], pd.DataFrame())

        def _normalize_df(df: pd.DataFrame):
            if isinstance(df, pd.DataFrame) and not df.empty:
                try:
                    out = df.copy()
                    out.columns = pd.to_datetime(out.columns, errors="ignore")
                    out = out.reindex(sorted(out.columns), axis=1)
                    return out
                except Exception:
                    return df
            return df

        q_inc, q_bal, q_cf = _normalize_df(q_inc), _normalize_df(q_bal), _normalize_df(q_cf)
        a_inc, a_bal, a_cf = _normalize_df(a_inc), _normalize_df(a_bal), _normalize_df(a_cf)

        # Selectors
        rev_q = extract_metric(q_inc, ["Total Revenue", "Operating Revenue", "Revenue", "Revenue From Contract With Customer Excluding Assessed Tax"])
        rev_a = extract_metric(a_inc, ["Total Revenue", "Operating Revenue", "Revenue", "Revenue From Contract With Customer Excluding Assessed Tax"])

        ni_q = extract_metric(q_inc, ["Net Income Common", "Net Income", "Net Income Applicable To Common Shares", "Net Income From Continuing Operation Net Minority Interest"])
        ni_a = extract_metric(a_inc, ["Net Income Common", "Net Income", "Net Income Applicable To Common Shares", "Net Income From Continuing Operation Net Minority Interest"])

        gross_profit_q = extract_metric(q_inc, ["Gross Profit"])
        gross_profit_a = extract_metric(a_inc, ["Gross Profit"])

        ebit_q = extract_metric(q_inc, ["EBIT", "Operating Income", "Operating Profit"])
        ebit_a = extract_metric(a_inc, ["EBIT", "Operating Income", "Operating Profit"])

        rev_ttm = get_ttm(rev_q, rev_a)
        ni_ttm = get_ttm(ni_q, ni_a)
        gross_profit_ttm = get_ttm(gross_profit_q, gross_profit_a)
        ebit_ttm = get_ttm(ebit_q, ebit_a)

        eq_q = extract_metric(q_bal, [
            "Stockholders Equity", "Total Equity", "Common Stock Equity", "Total Stockholder Equity",
            "Total Stockholders Equity", "Equity", "Shareholders Equity"
        ])
        eq_a = extract_metric(a_bal, [
            "Stockholders Equity", "Total Equity", "Common Stock Equity", "Total Stockholder Equity",
            "Total Stockholders Equity", "Equity", "Shareholders Equity"
        ])

        ast_q = extract_metric(q_bal, ["Total Assets", "Assets"])
        ast_a = extract_metric(a_bal, ["Total Assets", "Assets"])

        cast_q = extract_metric(q_bal, ["Current Assets", "Total Current Assets"])
        cast_a = extract_metric(a_bal, ["Current Assets", "Total Current Assets"])

        cliab_q = extract_metric(q_bal, ["Current Liabilities", "Total Current Liabilities"])
        cliab_a = extract_metric(a_bal, ["Current Liabilities", "Total Current Liabilities"])

        ldebt_q = extract_metric(q_bal, [
            "Long Term Debt", "Long Term Borrowings", "Non Current Debt",
            "Long Term Debt And Capital Lease Obligation", "Long Term Debt Noncurrent"
        ])
        ldebt_a = extract_metric(a_bal, [
            "Long Term Debt", "Long Term Borrowings", "Non Current Debt",
            "Long Term Debt And Capital Lease Obligation", "Long Term Debt Noncurrent"
        ])

        sdebt_q = extract_metric(q_bal, [
            "Current Debt", "Short Term Borrowings", "Current Portion Of Long Term Debt",
            "Current Portion Of Long Term Debt And Capital Lease Obligation"
        ])
        sdebt_a = extract_metric(a_bal, [
            "Current Debt", "Short Term Borrowings", "Current Portion Of Long Term Debt",
            "Current Portion Of Long Term Debt And Capital Lease Obligation"
        ])

        # Prefer audited annual balance-sheet values when available; fallback to quarterly.
        latest_eq = latest_value(eq_a if not eq_a.empty else eq_q)
        latest_ast = latest_value(ast_a if not ast_a.empty else ast_q)
        latest_ast_q = latest_value(ast_q)
        latest_cast = latest_value(cast_a if not cast_a.empty else cast_q)
        latest_cliab = latest_value(cliab_a if not cliab_a.empty else cliab_q)

        # Debt candidates
        ldebt_latest_a = latest_value(ldebt_a, 0.0)
        sdebt_latest_a = latest_value(sdebt_a, 0.0)
        ldebt_latest_q = latest_value(ldebt_q, 0.0)
        sdebt_latest_q = latest_value(sdebt_q, 0.0)

        total_debt_a = safe_num(ldebt_latest_a, 0.0) + safe_num(sdebt_latest_a, 0.0)
        total_debt_q = safe_num(ldebt_latest_q, 0.0) + safe_num(sdebt_latest_q, 0.0)
        total_debt_info = safe_num(info.get("totalDebt"), np.nan)

        total_debt = np.nan
        debt_source = "N/A"
        for label, candidate in [
            ("annual_calc", total_debt_a),
            ("quarter_calc", total_debt_q),
            ("info", total_debt_info),
        ]:
            if is_plausible_metric_value(candidate, "der", "general"):
                total_debt = safe_num(candidate)
                debt_source = label
                break

        ocf_q = extract_metric(q_cf, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"])
        ocf_a = extract_metric(a_cf, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"])

        capex_q = extract_metric(q_cf, ["Capital Expenditure", "Capital Expenditures", "Purchase Of Property Plant And Equipment", "Capital Expenditures Reported"])
        capex_a = extract_metric(a_cf, ["Capital Expenditure", "Capital Expenditures", "Purchase Of Property Plant And Equipment", "Capital Expenditures Reported"])

        ocf_ttm = get_ttm(ocf_q, ocf_a)
        capex_ttm = get_ttm(capex_q, capex_a)

        if not np.isnan(ocf_ttm) and not np.isnan(capex_ttm):
            fcf_ttm = ocf_ttm + capex_ttm
        else:
            fcf_ttm = safe_num(info.get("freeCashflow"), np.nan)

        price = safe_num(info.get("currentPrice"), safe_num(info.get("regularMarketPrice")))
        if np.isnan(price):
            price = safe_num(info.get("lastPrice"), np.nan)

        shares = safe_num(info.get("sharesOutstanding"), safe_num(info.get("impliedSharesOutstanding")))
        mcap = safe_num(info.get("marketCap"))
        if (np.isnan(mcap) or mcap <= 0) and not np.isnan(price) and not np.isnan(shares):
            mcap = price * shares

        # Statement-based per-share book value if possible
        book_value = safe_num(info.get("bookValue"))
        pb_annual = np.nan
        pb_quarter = np.nan
        pb_bvps = np.nan
        if not np.isnan(mcap) and latest_eq > 0:
            pb_annual = mcap / latest_eq

        # Quarter equity only as fallback if annual is absent
        latest_eq_q = latest_value(eq_q)
        if not np.isnan(mcap) and latest_eq_q > 0:
            pb_quarter = mcap / latest_eq_q

        if not np.isnan(price) and book_value > 0:
            pb_bvps = price / book_value

        pb_info = safe_num(info.get("priceToBook"))
        pb, pb_source = choose_first_plausible([
            ("calc_annual", pb_annual),
            ("calc_quarter", pb_quarter),
            ("price_book", pb_bvps),
            ("info", pb_info),
        ], "pbv", sector_bucket({"sector": info.get("sector", ""), "industry": info.get("industry", "")}))

        sector_bucket_name = sector_bucket({"sector": info.get("sector", ""), "industry": info.get("industry", "")})

        # Official website overrides (only when confidence is decent)
        official_confidence = safe_num((official_bundle or {}).get("confidence", 0.0), 0.0)
        if official_confidence >= 0.30 and isinstance(official_metrics, dict):
            official_rev = safe_num(official_metrics.get("rev_ttm"))
            official_ni = safe_num(official_metrics.get("ni_ttm"))
            official_eq = safe_num(official_metrics.get("equity"))
            official_ast = safe_num(official_metrics.get("assets"))
            official_cast = safe_num(official_metrics.get("current_assets"))
            official_cliab = safe_num(official_metrics.get("current_liabilities"))
            official_debt = safe_num(official_metrics.get("debt"))
            official_ocf = safe_num(official_metrics.get("ocf_ttm"))
            official_capex = safe_num(official_metrics.get("capex_ttm"))

            if official_rev > 0:
                rev_ttm = official_rev
            if official_ni != 0:
                ni_ttm = official_ni
            if official_eq > 0:
                latest_eq = official_eq
            if official_ast > 0:
                latest_ast = official_ast
            if official_cast > 0:
                latest_cast = official_cast
            if official_cliab > 0:
                latest_cliab = official_cliab
            if official_debt > 0:
                total_debt = official_debt
                debt_source = "official_web"
            if official_ocf != 0:
                ocf_ttm = official_ocf
            if official_capex != 0:
                capex_ttm = official_capex
            if not np.isnan(ocf_ttm) and not np.isnan(capex_ttm):
                fcf_ttm = ocf_ttm + capex_ttm

        # DER: prefer audited annual calc, then quarterly calc, then Yahoo info.
        calc_der_annual = np.nan
        calc_der_quarter = np.nan
        if latest_eq > 0 and total_debt_a > 0:
            calc_der_annual = total_debt_a / latest_eq
        if latest_eq_q > 0 and total_debt_q > 0:
            calc_der_quarter = total_debt_q / latest_eq_q

        info_der = normalize_percent_like(info.get("debtToEquity"))

        der, der_source = choose_first_plausible([
            ("calc_annual", calc_der_annual),
            ("calc_quarter", calc_der_quarter),
            ("info", info_der),
        ], "der", sector_bucket_name)

        cr = safe_num(info.get("currentRatio"))
        if (np.isnan(cr) or cr <= 0) and latest_cliab > 0:
            cr = latest_cast / latest_cliab

        # Profitability ratios: use statement-derived values first, info only as fallback.
        roe_calc = np.nan
        roa_calc = np.nan
        npm_calc = np.nan
        if latest_eq > 0:
            roe_calc = (ni_ttm / latest_eq) * 100.0
        elif latest_eq_q > 0:
            roe_calc = (ni_ttm / latest_eq_q) * 100.0

        if latest_ast > 0:
            roa_calc = (ni_ttm / latest_ast) * 100.0
        elif latest_ast_q > 0:
            roa_calc = (ni_ttm / latest_ast_q) * 100.0

        if rev_ttm > 0:
            npm_calc = (ni_ttm / rev_ttm) * 100.0

        roe_info = normalize_percent_like(info.get("returnOnEquity"))
        roa_info = normalize_percent_like(info.get("returnOnAssets"))
        npm_info = normalize_percent_like(info.get("profitMargins"))

        roe, roe_source = choose_first_plausible([
            ("calc", roe_calc),
            ("info", roe_info),
        ], "roe", sector_bucket_name)

        roa, roa_source = choose_first_plausible([
            ("calc", roa_calc),
            ("info", roa_info),
        ], "roa", sector_bucket_name)

        npm, npm_source = choose_first_plausible([
            ("calc", npm_calc),
            ("info", npm_info),
        ], "npm", sector_bucket_name)

        pe = safe_num(info.get("trailingPE"))
        if (np.isnan(pe) or pe <= 0) and not np.isnan(mcap) and ni_ttm > 0:
            pe = mcap / ni_ttm
        if not is_plausible_metric_value(pe, "pe", sector_bucket_name):
            pe = np.nan

        peg = safe_num(info.get("pegRatio"))
        if np.isnan(peg) or peg <= 0 or peg > 50:
            peg = np.nan

        rev_growth_yoy = growth_from_series(rev_q, rev_a)
        ni_growth_yoy = growth_from_series(ni_q, ni_a)
        ocf_growth_yoy = growth_from_series(ocf_q, ocf_a)

        gross_margin = np.nan
        ebit_margin = np.nan
        if not np.isnan(rev_ttm) and rev_ttm > 0:
            if not np.isnan(gross_profit_ttm):
                gross_margin = (gross_profit_ttm / rev_ttm) * 100.0
            if not np.isnan(ebit_ttm):
                ebit_margin = (ebit_ttm / rev_ttm) * 100.0

        # Data quality / source audit notes for debugging
        data_quality_notes = []
        if not np.isnan(pb_annual):
            data_quality_notes.append(f"PB_annual={pb_annual:.2f}")
        if not np.isnan(pb_quarter):
            data_quality_notes.append(f"PB_quarter={pb_quarter:.2f}")
        if not np.isnan(info_der):
            data_quality_notes.append(f"DER_info={info_der:.2f}")
        if not np.isnan(calc_der_annual):
            data_quality_notes.append(f"DER_annual={calc_der_annual:.2f}")
        if not np.isnan(calc_der_quarter):
            data_quality_notes.append(f"DER_quarter={calc_der_quarter:.2f}")
        if not np.isnan(roe_calc):
            data_quality_notes.append(f"ROE_calc={roe_calc:.2f}")
        if not np.isnan(roa_calc):
            data_quality_notes.append(f"ROA_calc={roa_calc:.2f}")

        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "currency": info.get("currency", "N/A"),
            "price": price,
            "mcap": mcap,
            "pe": pe,
            "pb": pb,
            "peg": peg,
            "roe": roe,
            "roa": roa,
            "npm": npm,
            "der": der,
            "cr": cr,
            "ocf": ocf_ttm,
            "fcf": fcf_ttm,
            "rev_ttm": rev_ttm,
            "ni_ttm": ni_ttm,
            "debt": total_debt,
            "equity": latest_eq,
            "assets": latest_ast,
            "rev_growth_yoy": rev_growth_yoy,
            "ni_growth_yoy": ni_growth_yoy,
            "ocf_growth_yoy": ocf_growth_yoy,
            "gross_margin": gross_margin,
            "ebit_margin": ebit_margin,
            "sector_bucket": sector_bucket_name,
            "pb_source": pb_source,
            "der_source": der_source,
            "roe_source": roe_source,
            "roa_source": roa_source,
            "npm_source": npm_source,
            "debt_source": debt_source,
            "official_source_type": (official_bundle or {}).get("source_type", "N/A"),
            "official_source_urls": (official_bundle or {}).get("source_urls", []),
            "official_confidence": official_confidence if 'official_confidence' in locals() else 0.0,
            "resolved_website": resolved_web,
            "website_resolution_source": web_resolve_bundle.get("source", "none"),
            "website_resolution_score": safe_num(web_resolve_bundle.get("score", 0.0), 0.0),
            "website_candidates": web_resolve_bundle.get("candidates", []),
            "data_quality": " | ".join(data_quality_notes[:8]),
        }
    except Exception:
        return {}

# ------------------------------------------------------------------------------
# 5. TECHNICAL ANALYSIS ENGINE (SMC / ICT + TRADER LOGIC)
# ------------------------------------------------------------------------------
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_technicals(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().dropna(subset=["Close"])
    if len(d) < 20:
        return pd.DataFrame()

    d["SMA20"] = d["Close"].rolling(20).mean()
    d["SMA50"] = d["Close"].rolling(50).mean()
    d["SMA200"] = d["Close"].rolling(200).mean()
    d["RSI14"] = calc_rsi(d["Close"], 14)
    d["ATR14"] = calc_atr(d, 14)
    d["VolSMA20"] = d["Volume"].rolling(20).mean()

    low14 = d["Low"].rolling(14).min()
    high14 = d["High"].rolling(14).max()
    d["StochK"] = 100 * (d["Close"] - low14) / (high14 - low14).replace(0, np.nan)

    d["Swing_High"] = (d["High"] == d["High"].rolling(5, center=True).max())
    d["Swing_Low"] = (d["Low"] == d["Low"].rolling(5, center=True).min())

    d["Breakout_20D"] = d["Close"] > d["High"].rolling(20).max().shift(1)
    d["Breakdown_20D"] = d["Close"] < d["Low"].rolling(20).min().shift(1)
    d["Volume_Ratio"] = d["Volume"] / d["VolSMA20"]

    n = len(d)
    bull_bos, bull_choch, bull_fvg, bull_sweep, order_block = [False] * n, [False] * n, [False] * n, [False] * n, [False] * n
    last_sh, last_sl = np.nan, np.nan
    trend = 0

    h_arr = d["High"].values
    l_arr = d["Low"].values
    c_arr = d["Close"].values
    o_arr = d["Open"].values
    sh_arr = d["Swing_High"].values
    sl_arr = d["Swing_Low"].values

    for i in range(2, n):
        if l_arr[i] > h_arr[i - 2]:
            bull_fvg[i] = True

        if sh_arr[i - 1]:
            last_sh = h_arr[i - 1]
        if sl_arr[i - 1]:
            last_sl = l_arr[i - 1]

        if not np.isnan(last_sh) and c_arr[i] > last_sh and c_arr[i - 1] <= last_sh:
            bull_bos[i] = True
            if trend <= 0:
                bull_choch[i] = True
            trend = 1

            for j in range(i - 1, max(0, i - 10), -1):
                if c_arr[j] < o_arr[j]:
                    order_block[j] = True
                    break

        if not np.isnan(last_sl) and l_arr[i] < last_sl and c_arr[i] > last_sl:
            bull_sweep[i] = True

    d["Bullish_BOS"] = bull_bos
    d["Bullish_ChoCH"] = bull_choch
    d["Bullish_FVG"] = bull_fvg
    d["Bullish_OB"] = order_block
    d["Liquidity_Sweep"] = bull_sweep

    recent_l = d["Low"].rolling(20).min()
    recent_h = d["High"].rolling(20).max()
    d["Discount"] = d["Close"] < ((recent_h + recent_l) / 2)
    d["Near_Support"] = d["Close"] <= (recent_l * 1.05)
    d["Near_Resistance"] = d["Close"] >= (recent_h * 0.95)
    d["Above_SMA20"] = d["Close"] > d["SMA20"]
    d["Above_SMA50"] = d["Close"] > d["SMA50"]
    d["Trend_Strong"] = (d["SMA20"] > d["SMA50"]) & (d["Close"] > d["SMA20"])
    d["Near_52W_High"] = d["Close"] >= d["Close"].rolling(252, min_periods=60).max() * 0.90
    d["Near_52W_Low"] = d["Close"] <= d["Close"].rolling(252, min_periods=60).min() * 1.10

    return d


# ------------------------------------------------------------------------------
# 6. SCORING ENGINE
# ------------------------------------------------------------------------------
def score_quality(fund: dict) -> float:
    score = 0.0
    if not fund:
        return score

    roe = safe_num(fund.get("roe"))
    roa = safe_num(fund.get("roa"))
    npm = safe_num(fund.get("npm"))
    der = safe_num(fund.get("der"))
    cr = safe_num(fund.get("cr"))
    fcf = safe_num(fund.get("fcf"), 0)
    ocf = safe_num(fund.get("ocf"), 0)
    pe = safe_num(fund.get("pe"))
    sector_bucket_name = str(fund.get("sector_bucket", "general")).lower()

    if roe >= 15:
        score += 18
    elif roe >= 10:
        score += 12
    elif roe >= 5:
        score += 6

    if roa >= 8:
        score += 10
    elif roa >= 4:
        score += 6

    if npm >= 10:
        score += 10
    elif npm >= 5:
        score += 6

    if cr >= 1.5:
        score += 8
    elif cr >= 1.0:
        score += 4

    if fcf > 0:
        score += 10
    if ocf > 0:
        score += 8

    if sector_bucket_name != "financial":
        if der <= 1.0:
            score += 8
        elif der <= 1.5:
            score += 4
    else:
        # Untuk financial, leverage dinilai lebih longgar.
        if der <= 10:
            score += 4

    if 0 < pe <= 15:
        score += 10
    elif 15 < pe <= 25:
        score += 5

    return min(score, 100)


def score_growth(fund: dict) -> float:
    score = 0.0
    if not fund:
        return score

    rev_growth = safe_num(fund.get("rev_growth_yoy"))
    ni_growth = safe_num(fund.get("ni_growth_yoy"))
    ocf_growth = safe_num(fund.get("ocf_growth_yoy"))
    rev_ttm = safe_num(fund.get("rev_ttm"))
    ni_ttm = safe_num(fund.get("ni_ttm"))
    gross_margin = safe_num(fund.get("gross_margin"))
    ebit_margin = safe_num(fund.get("ebit_margin"))
    roe = safe_num(fund.get("roe"))
    npm = safe_num(fund.get("npm"))
    fcf = safe_num(fund.get("fcf"), 0)

    if not np.isnan(rev_growth):
        if rev_growth >= 25:
            score += 20
        elif rev_growth >= 15:
            score += 16
        elif rev_growth >= 8:
            score += 10

    if not np.isnan(ni_growth):
        if ni_growth >= 25:
            score += 18
        elif ni_growth >= 15:
            score += 14
        elif ni_growth >= 8:
            score += 8

    if not np.isnan(ocf_growth):
        if ocf_growth >= 20:
            score += 10
        elif ocf_growth >= 10:
            score += 6

    if rev_ttm > 0:
        score += 6
    if ni_ttm > 0:
        score += 6
    if fcf > 0:
        score += 8

    if not np.isnan(gross_margin) and gross_margin >= 20:
        score += 6
    if not np.isnan(ebit_margin) and ebit_margin >= 10:
        score += 6

    if roe >= 15:
        score += 8
    if npm >= 10:
        score += 4

    return min(score, 100)


def score_value(fund: dict) -> float:
    score = 0.0
    if not fund:
        return score

    pe = safe_num(fund.get("pe"))
    pb = safe_num(fund.get("pb"))
    peg = safe_num(fund.get("peg"))
    roe = safe_num(fund.get("roe"))
    der = safe_num(fund.get("der"))
    fcf = safe_num(fund.get("fcf"), 0)
    sector_bucket_name = str(fund.get("sector_bucket", "general")).lower()

    if sector_bucket_name == "financial":
        if 0 < pb <= 1.5:
            score += 20
        elif 1.5 < pb <= 2.5:
            score += 14
        elif 2.5 < pb <= 4:
            score += 8
        if roe >= 15:
            score += 16
        elif roe >= 10:
            score += 10
        if fcf > 0:
            score += 8
    else:
        if 0 < pe <= 10:
            score += 18
        elif 10 < pe <= 15:
            score += 14
        elif 15 < pe <= 20:
            score += 8

        if 0 < pb <= 1.5:
            score += 12
        elif 1.5 < pb <= 3:
            score += 8

        if not np.isnan(peg) and peg > 0:
            if peg <= 1.0:
                score += 12
            elif peg <= 1.5:
                score += 8

        if roe >= 15:
            score += 10
        elif roe >= 10:
            score += 6

        if der <= 1.0:
            score += 6
        elif der <= 1.5:
            score += 3

        if fcf > 0:
            score += 8

    return min(score, 100)


def score_smc(tech: pd.DataFrame) -> float:
    score = 0.0
    if tech is None or tech.empty:
        return score

    last = tech.iloc[-1]
    if bool(last.get("Bullish_ChoCH")):
        score += 22
    if bool(last.get("Liquidity_Sweep")):
        score += 14
    if bool(last.get("Bullish_BOS")):
        score += 10
    if bool(last.get("Bullish_FVG")):
        score += 8
    if bool(last.get("Bullish_OB")):
        score += 10
    if bool(last.get("Discount")):
        score += 8
    if bool(last.get("Near_Support")):
        score += 8

    rsi = safe_num(last.get("RSI14"))
    stoch = safe_num(last.get("StochK"))
    if (rsi <= 35) or (stoch <= 20):
        score += 10
    elif 35 < rsi <= 60:
        score += 6

    return min(score, 100)


def score_trader(tech: pd.DataFrame) -> float:
    """Logika trader umum: trend, breakout, volume, dan risk-reward."""
    score = 0.0
    if tech is None or tech.empty:
        return score

    last = tech.iloc[-1]
    close = safe_num(last.get("Close"))
    sma20 = safe_num(last.get("SMA20"))
    sma50 = safe_num(last.get("SMA50"))
    sma200 = safe_num(last.get("SMA200"))
    rsi = safe_num(last.get("RSI14"))
    atr = safe_num(last.get("ATR14"))
    vol_ratio = safe_num(last.get("Volume_Ratio"))

    if close > sma20:
        score += 10
    if close > sma50:
        score += 8
    if not np.isnan(sma20) and not np.isnan(sma50) and sma20 > sma50:
        score += 12
    if not np.isnan(sma200) and close > sma200:
        score += 8

    if bool(last.get("Breakout_20D")):
        score += 16
    if bool(last.get("Near_Resistance")) and bool(last.get("Breakout_20D")):
        score += 6

    if not np.isnan(vol_ratio):
        if vol_ratio >= 1.8:
            score += 12
        elif vol_ratio >= 1.3:
            score += 8
        elif vol_ratio >= 1.0:
            score += 4

    if 45 <= rsi <= 65:
        score += 12
    elif 35 <= rsi < 45:
        score += 8
    elif rsi > 70:
        score += 2  # overbought, masih bisa lanjut namun tidak ideal

    if not np.isnan(atr) and atr > 0:
        recent_low = safe_num(last.get("Low"))
        rr = np.nan
        # target konservatif: 2 x ATR dari close, stop: 1 x ATR di bawah low
        stop = recent_low - atr
        target = close + (2 * atr)
        risk = close - stop
        reward = target - close
        if risk > 0:
            rr = reward / risk
        if not np.isnan(rr):
            if rr >= 3:
                score += 10
            elif rr >= 2:
                score += 8
            elif rr >= 1.5:
                score += 4

    if bool(last.get("Trend_Strong")):
        score += 10
    if bool(last.get("Near_Support")):
        score += 6

    return min(score, 100)


def score_andry_hakim_style(tech: pd.DataFrame, fund: dict) -> float:
    """Heuristik momentum/breakout + disiplin cut-loss. Bukan klaim metode asli."""
    score = 0.0
    if tech is None or tech.empty:
        return score
    last = tech.iloc[-1]

    close = safe_num(last.get("Close"))
    sma20 = safe_num(last.get("SMA20"))
    sma50 = safe_num(last.get("SMA50"))
    rsi = safe_num(last.get("RSI14"))
    vol_ratio = safe_num(last.get("Volume_Ratio"))
    atr = safe_num(last.get("ATR14"))

    if close > sma20 > sma50:
        score += 30
    elif close > sma20:
        score += 18

    if bool(last.get("Breakout_20D")):
        score += 20
    if not np.isnan(vol_ratio) and vol_ratio >= 1.5:
        score += 15
    if 45 <= rsi <= 70:
        score += 10
    if bool(last.get("Bullish_BOS")) or bool(last.get("Bullish_ChoCH")):
        score += 15

    # Disiplin risk management: tidak terlalu jauh dari support / ATR masih sehat
    if not np.isnan(atr) and atr > 0:
        if (safe_num(last.get("Close")) - safe_num(last.get("Low"))) / atr <= 2.5:
            score += 10

    # Kualitas fundamental sebagai filter
    if safe_num(fund.get("roe")) >= 10:
        score += 5

    return min(score, 100)


def score_hengky_adinata_style(tech: pd.DataFrame, fund: dict) -> float:
    """Heuristik swing/accumulation + support-resistance + selective entry. Bukan klaim metode asli."""
    score = 0.0
    if tech is None or tech.empty:
        return score
    last = tech.iloc[-1]

    if bool(last.get("Near_Support")):
        score += 18
    if bool(last.get("Discount")):
        score += 12
    if bool(last.get("Liquidity_Sweep")):
        score += 15
    if bool(last.get("Bullish_ChoCH")):
        score += 15
    if bool(last.get("Bullish_OB")):
        score += 12
    if bool(last.get("Bullish_FVG")):
        score += 8

    rsi = safe_num(last.get("RSI14"))
    stoch = safe_num(last.get("StochK"))
    if rsi < 40 or stoch < 25:
        score += 10
    if 40 <= rsi <= 60:
        score += 6

    # Filter kualitas supaya tidak asal buy saham sakit
    if safe_num(fund.get("fcf"), 0) > 0:
        score += 10
    if safe_num(fund.get("cr")) >= 1:
        score += 5

    return min(score, 100)


FOCUS_WEIGHTS = {
    "Komprehensif": {"quality": 0.20, "growth": 0.20, "value": 0.20, "trader": 0.18, "smc": 0.14, "andry": 0.04, "hengky": 0.04},
    "Trader": {"quality": 0.10, "growth": 0.08, "value": 0.07, "trader": 0.28, "smc": 0.18, "andry": 0.17, "hengky": 0.12},
    "Growth Investing": {"quality": 0.25, "growth": 0.35, "value": 0.08, "trader": 0.12, "smc": 0.08, "andry": 0.06, "hengky": 0.06},
    "Value Investing": {"quality": 0.25, "growth": 0.10, "value": 0.35, "trader": 0.12, "smc": 0.08, "andry": 0.05, "hengky": 0.05},
}


def combine_score(scores: Dict[str, float], focus: str = "Komprehensif", bandar_mod: float = 0.0) -> float:
    weights = FOCUS_WEIGHTS.get(focus, FOCUS_WEIGHTS["Komprehensif"])
    total = 0.0
    for k, w in weights.items():
        total += safe_num(scores.get(k), 0.0) * w
    total += bandar_mod
    return float(min(100, max(0, total)))


def bandarmology_modifier(label: str) -> float:
    label = str(label).lower()
    if "big" in label:
        return 15.0
    if "small" in label:
        return 7.5
    if "dist" in label:
        return -15.0
    return 0.0


# ------------------------------------------------------------------------------
# 7. PARALLEL SCAN WORKERS
# ------------------------------------------------------------------------------
def worker_scan_ticker(t: str, focus: str = "Komprehensif"):
    try:
        df = fetch_price(t)
        if df.empty or len(df) < 20:
            return None

        tdf = calculate_technicals(df)
        if tdf.empty:
            return None

        fund = fetch_fundamentals(
            t,
            use_official_web_validation=st.session_state.get("enable_official_web_validation", True),
            website_override=st.session_state.get("official_website_override", ""),
            auto_discover_website=st.session_state.get("enable_auto_website_discovery", True),
        )
        if not fund:
            return None

        s_quality = score_quality(fund)
        s_growth = score_growth(fund)
        s_value = score_value(fund)
        s_trader = score_trader(tdf)
        s_smc = score_smc(tdf)
        s_andry = score_andry_hakim_style(tdf, fund)
        s_hengky = score_hengky_adinata_style(tdf, fund)

        scores = {
            "quality": s_quality,
            "growth": s_growth,
            "value": s_value,
            "trader": s_trader,
            "smc": s_smc,
            "andry": s_andry,
            "hengky": s_hengky,
        }
        final_score = combine_score(scores, focus=focus, bandar_mod=0.0)
        last = tdf.iloc[-1]

        return {
            "Ticker": strip_suffix(t),
            "Harga": float(last["Close"]),
            "Final Score": float(final_score),
            "Kategori": investment_status(final_score),
            "PE (x)": fund.get("pe", np.nan),
            "PBV (x)": fund.get("pb", np.nan),
            "PEG (x)": fund.get("peg", np.nan),
            "ROE (%)": fund.get("roe", np.nan),
            "ROA (%)": fund.get("roa", np.nan),
            "NPM (%)": fund.get("npm", np.nan),
            "DER (x)": fund.get("der", np.nan),
            "Current Ratio (x)": fund.get("cr", np.nan),
            "FCF": fund.get("fcf", np.nan),
            "OCF": fund.get("ocf", np.nan),
            "Revenue Growth (%)": fund.get("rev_growth_yoy", np.nan),
            "Net Income Growth (%)": fund.get("ni_growth_yoy", np.nan),
            "OCF Growth (%)": fund.get("ocf_growth_yoy", np.nan),
            "RSI": round(float(last["RSI14"]), 1) if pd.notna(last["RSI14"]) else np.nan,
            "Stoch K": round(float(last["StochK"]), 1) if pd.notna(last["StochK"]) else np.nan,
            "SMC Reversal": "✅ Ya" if bool(last.get("Bullish_ChoCH")) or bool(last.get("Liquidity_Sweep")) else "❌ Tidak",
            "Official Check": "✅" if fund.get("official_confidence", 0) and fund.get("official_confidence", 0) >= 0.30 else "⏳",
            "Website": fund.get("resolved_website", "N/A"),
            "Data Source": fund.get("official_source_type", "yfinance"),
            "Quality Score": round(float(s_quality), 1),
            "Growth Score": round(float(s_growth), 1),
            "Value Score": round(float(s_value), 1),
            "Trader Score": round(float(s_trader), 1),
            "SMC Score": round(float(s_smc), 1),
            "Andry Style": round(float(s_andry), 1),
            "Hengky Style": round(float(s_hengky), 1),
        }
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def scan_universe_cached(tickers: Tuple[str, ...], focus: str) -> pd.DataFrame:
    results = []
    if not tickers:
        return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=SCAN_MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(worker_scan_ticker, t, focus): t for t in tickers}
        for future in as_completed(future_to_ticker):
            res = future.result()
            if res is not None:
                results.append(res)

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)
    for col in [
        "Ticker", "Harga", "Final Score", "Kategori", "PE (x)", "PBV (x)", "PEG (x)",
        "ROE (%)", "ROA (%)", "NPM (%)", "DER (x)", "Current Ratio (x)",
        "FCF", "OCF", "Revenue Growth (%)", "Net Income Growth (%)", "OCF Growth (%)",
        "RSI", "Stoch K", "SMC Reversal", "Quality Score", "Growth Score", "Value Score",
        "Trader Score", "SMC Score", "Andry Style", "Hengky Style",
    ]:
        if col not in df_res.columns:
            df_res[col] = np.nan

    return df_res.sort_values("Final Score", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------------------
# 8. STREAMLIT CONTROL PANEL SIDEBAR
# ------------------------------------------------------------------------------
st.title("Smart Alpha Screener v17.0")
st.caption("Screening IHSG: fundamental, growth, value, trader logic, dan smart money dalam satu alur.")

with st.sidebar:
    st.header("🎯 Smart Alpha Kontrol")

    st.checkbox("Validasi fundamental via situs perusahaan", value=True, key="enable_official_web_validation")
    st.checkbox("Auto-cari website resmi dari ticker", value=True, key="enable_auto_website_discovery")
    st.text_input("Override website perusahaan (opsional)", value="", placeholder="https://...", key="official_website_override")

    current_mode = st.radio(
        "Pilih Mode",
        ["Light Scan Multi-Saham", "Deep Dive Target"],
        index=0 if st.session_state.app_mode == "Light Scan Multi-Saham" else 1,
    )
    st.session_state.app_mode = current_mode
    st.divider()

    st.session_state.strategy_focus = st.selectbox(
        "Fokus Strategi",
        ["Komprehensif", "Trader", "Growth Investing", "Value Investing"],
        index=["Komprehensif", "Trader", "Growth Investing", "Value Investing"].index(st.session_state.strategy_focus)
        if st.session_state.strategy_focus in ["Komprehensif", "Trader", "Growth Investing", "Value Investing"] else 0,
    )

    if st.session_state.app_mode == "Light Scan Multi-Saham":
        st.markdown("### Universe Scan Options")

        custom_input = st.text_input(
            "Tambah Ticker Kustom Diluar Daftar (Pisahkan dengan koma)",
            placeholder="Contoh: MLBI, YULE, VRNA",
            key="custom_tickers_input",
        ).strip().upper()

        custom_list = []
        if custom_input:
            custom_list = [ensure_yf_ticker(x.strip()) for x in custom_input.split(",") if x.strip()]

        st.session_state.universe_search = st.text_input(
            "Filter nama / kode ticker",
            value=st.session_state.universe_search,
            placeholder="Contoh: bank, tlkm, bbri",
        ).strip().upper()

        full_universe_pool = unique_keep_order(ALL_IHSG + custom_list)

        filtered_universe = [
            t for t in full_universe_pool
            if st.session_state.universe_search in strip_suffix(t).upper()
        ] if st.session_state.universe_search else full_universe_pool

        st.checkbox("Pilih semua hasil filter", value=st.session_state.select_all_filtered, key="select_all_filtered")

        default_selection = filtered_universe if st.session_state.select_all_filtered else st.session_state.active_universe
        default_selection = [t for t in default_selection if t in filtered_universe]

        for cl in custom_list:
            if cl in filtered_universe and cl not in default_selection:
                default_selection.append(cl)

        selected_tickers = st.multiselect(
            "Pilih / ketik ticker aktif",
            options=filtered_universe,
            default=default_selection,
            format_func=lambda x: strip_suffix(x),
        )

        st.session_state.active_universe = selected_tickers[:MAX_SCAN_UNIVERSE]
        if len(selected_tickers) > MAX_SCAN_UNIVERSE:
            st.warning(f"Jumlah scan dibatasi ke {MAX_SCAN_UNIVERSE} item teratas demi menjaga kestabilan API.")

        st.session_state.selected_status_filter = st.selectbox(
            "Filter status hasil",
            ["Semua", "LAYAK BELI", "WATCHLIST", "NETRAL", "HINDARI"],
            index=["Semua", "LAYAK BELI", "WATCHLIST", "NETRAL", "HINDARI"].index(st.session_state.selected_status_filter)
            if st.session_state.selected_status_filter in ["Semua", "LAYAK BELI", "WATCHLIST", "NETRAL", "HINDARI"]
            else 0,
        )

        st.session_state.min_final_score = st.slider(
            "Minimum Final Score",
            0, 100, int(st.session_state.min_final_score), 1,
        )

        run_scan = st.button("🚀 Jalankan Scan Paralel", use_container_width=True)

        if run_scan:
            if not st.session_state.active_universe:
                st.error("Pilih atau masukkan minimal 1 saham untuk di-scan.")
            else:
                with st.spinner("Menarik data fundamental, harga, dan struktur pasar..."):
                    st.session_state.cached_scan_results = scan_universe_cached(
                        tuple(st.session_state.active_universe),
                        st.session_state.strategy_focus,
                    )

        st.divider()
        st.caption("Mode deep dive akan terbuka otomatis atau via tombol sorotan emiten.")

    else:
        st.markdown("### Target Analisis")
        target_input = st.text_input(
            "Masukkan kode ticker target",
            value=st.session_state.target_ticker,
        ).strip().upper()
        st.session_state.target_ticker = target_input

        st.link_button("🌐 Data Akumulasi NeoBDM", "https://neobdm.tech/")
        st.link_button("🌐 Broker Summary Stockbit", f"https://stockbit.com/symbol/{target_input}")

        bandar_status = st.radio(
            "Validasi Silent Accumulation Smart Money",
            ["Big Accumulation (+15)", "Small Accumulation (+7.5)", "Netral (0)", "Distribution (-15)"],
            index=2,
        )

        execute_deep_dive = st.button("⚙️ Eksekusi Analisis Mendalam", use_container_width=True)
        st.divider()
        st.caption("Deep dive menampilkan struktur harga SMC, finansial growth, dan chart.")


# ------------------------------------------------------------------------------
# 9. LAYOUT: LIGHT SCAN MULTI-SAHAM VIEW
# ------------------------------------------------------------------------------
if st.session_state.app_mode == "Light Scan Multi-Saham":
    st.subheader("Dashboard Multi-Saham Light Scan")

    res_df = st.session_state.cached_scan_results

    if res_df is not None and not res_df.empty:
        if st.session_state.selected_status_filter != "Semua":
            res_df = res_df[res_df["Kategori"] == st.session_state.selected_status_filter].copy()

        if st.session_state.min_final_score > 0:
            res_df = res_df[res_df["Final Score"] >= st.session_state.min_final_score].copy()

        if res_df.empty:
            st.warning("Tidak ada hasil saham yang cocok dengan filter indikator saat ini.")
        else:
            top_3 = res_df.head(3)
            st.markdown("### 🏆 Top 3 Rekomendasi Portofolio")
            c1, c2, c3 = st.columns(3)
            cols = [c1, c2, c3]

            for idx, (_, row) in enumerate(top_3.iterrows()):
                with cols[idx]:
                    st.metric(row["Ticker"], f'{row["Final Score"]:.1f}', row["Kategori"])
                    st.write(f"**Harga:** {fmt_num(row['Harga'])}")
                    st.write(f"**PE / PBV:** {fmt_num(row['PE (x)'])}x / {fmt_num(row['PBV (x)'])}x")
                    st.write(f"**RSI / Stoch:** {fmt_num(row['RSI'])} / {fmt_num(row['Stoch K'])}")
                    st.write(f"**SMC:** {row['SMC Reversal']}")

            st.divider()

            csv_bytes = res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download hasil scan (.CSV)",
                data=csv_bytes,
                file_name="smart_alpha_scan_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown("### 🔍 Hasil Emiten Sorotan")
            top_n = min(20, len(res_df))
            for idx, row in res_df.head(top_n).iterrows():
                with st.container(border=True):
                    colA, colB, colC, colD, colE = st.columns([1.4, 1.3, 1.6, 2.0, 1.2])
                    colA.markdown(f"**{row['Ticker']}**")
                    colB.metric("Score", f"{row['Final Score']:.1f}")
                    colC.write(f"**Kategori:** {row['Kategori']}\n\n**RSI:** {row['RSI']}\n**Stoch:** {row['Stoch K']}")
                    colD.write(
                        f"**PE:** {fmt_num(row['PE (x)'])}x  \n"
                        f"**PBV:** {fmt_num(row['PBV (x)'])}x  \n"
                        f"**ROE:** {pct(row['ROE (%)'])}  \n"
                        f"**DER:** {fmt_num(row['DER (x)'])}x  \n"
                        f"**Growth Rev:** {pct(row['Revenue Growth (%)'])}"
                    )
                    with colE:
                        if st.button(f"Deep Dive {row['Ticker']}", key=f"dd_btn_{row['Ticker']}_{idx}"):
                            st.session_state.target_ticker = str(row["Ticker"])
                            st.session_state.app_mode = "Deep Dive Target"
                            st.session_state["auto_deep_dive"] = True
                            st.rerun()

            st.divider()
            st.markdown("### Statistik Ringkas Universe")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Jumlah Tersaring", f"{len(res_df)}")
            c2.metric("Rata-rata Score", f"{res_df['Final Score'].mean():.1f}")
            c3.metric("Median Score", f"{res_df['Final Score'].median():.1f}")
            c4.metric("Jumlah LAYAK BELI", f"{int((res_df['Kategori'] == 'LAYAK BELI').sum())}")
    else:
        st.info("Masukkan daftar ticker pilihan Anda di sidebar, lalu jalankan scan paralel.")


# ------------------------------------------------------------------------------
# 10. LAYOUT: DEEP DIVE TARGET SAHAM VIEW
# ------------------------------------------------------------------------------
elif st.session_state.app_mode == "Deep Dive Target":
    target = ensure_yf_ticker(st.session_state.target_ticker)

    should_run_analysis = ("execute_deep_dive" in locals() and execute_deep_dive) or st.session_state.get("auto_deep_dive", False)

    if should_run_analysis:
        st.session_state["auto_deep_dive"] = False

        with st.spinner("Menarik data harga, fundamental komprehensif, dan memetakan struktur pasar..."):
            df = fetch_price(target)
            fund = fetch_fundamentals(
                target,
                use_official_web_validation=st.session_state.get("enable_official_web_validation", True),
                website_override=st.session_state.get("official_website_override", ""),
            )

            if df.empty or not fund:
                st.error("Gagal menarik data spesifik emiten. Pastikan kode ticker valid di Yahoo Finance.")
            else:
                tdf = calculate_technicals(df)

                if tdf.empty:
                    st.error("Gagal Melakukan Analisis: data teknikal/riwayat bursa tidak mencukupi (minimal diperlukan 20 bar aktif).")
                else:
                    active_bandar = bandar_status if "bandar_status" in locals() else "Netral (0)"
                    bandar_mod = bandarmology_modifier(active_bandar)

                    s_quality = score_quality(fund)
                    s_growth = score_growth(fund)
                    s_value = score_value(fund)
                    s_trader = score_trader(tdf)
                    s_smc = score_smc(tdf)
                    s_andry = score_andry_hakim_style(tdf, fund)
                    s_hengky = score_hengky_adinata_style(tdf, fund)

                    scores = {
                        "quality": s_quality,
                        "growth": s_growth,
                        "value": s_value,
                        "trader": s_trader,
                        "smc": s_smc,
                        "andry": s_andry,
                        "hengky": s_hengky,
                    }
                    final_score = combine_score(scores, focus=st.session_state.strategy_focus, bandar_mod=bandar_mod)
                    rating = investment_status(final_score)

                    st.subheader(f"Deep Dive Target: {strip_suffix(target)} - {fund.get('name', target)}")

                    if final_score >= 80:
                        st.success(f"KEPUTUSAN INVESTASI: {rating}. Adjusted Score: {final_score:.1f}/100")
                    elif final_score >= 65:
                        st.info(f"KEPUTUSAN INVESTASI: {rating}. Adjusted Score: {final_score:.1f}/100")
                    elif final_score >= 50:
                        st.warning(f"KEPUTUSAN INVESTASI: {rating}. Adjusted Score: {final_score:.1f}/100")
                    else:
                        st.error(f"KEPUTUSAN INVESTASI: {rating}. Adjusted Score: {final_score:.1f}/100")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Adjusted Final Score", f"{final_score:.1f}", f"Bandar Mod: {bandar_mod:+.1f}")
                    c2.metric("Skor Fundamental", f"{(s_quality + s_growth + s_value) / 3:.1f}")
                    c3.metric("Skor Teknikal", f"{(s_trader + s_smc) / 2:.1f}")
                    c4.metric("Harga Saat Ini", fmt_num(fund.get("price")))

                    st.caption(status_reason(final_score, fund, tdf, scores))
                    st.divider()

                    tab1, tab2, tab3 = st.tabs(["Fundamental & Growth", "Trader / Value / SMC", "Interactive Technical Chart"])

                    with tab1:
                        a, b = st.columns(2)
                        with a:
                            st.markdown("#### Valuasi & Pertumbuhan")
                            st.write(f"**Revenue TTM:** {fmt_num(fund.get('rev_ttm'))}")
                            st.write(f"**Net Income TTM:** {fmt_num(fund.get('ni_ttm'))}")
                            st.write(f"**Operating Cash Flow:** {fmt_num(fund.get('ocf'))}")
                            st.write(f"**Free Cash Flow:** {fmt_num(fund.get('fcf'))}")
                            st.write(f"**Revenue Growth YoY:** {pct(fund.get('rev_growth_yoy'))}")
                            st.write(f"**Net Income Growth YoY:** {pct(fund.get('ni_growth_yoy'))}")
                            st.write(f"**OCF Growth YoY:** {pct(fund.get('ocf_growth_yoy'))}")
                            st.write(f"**PE TTM:** {fmt_num(fund.get('pe'))}x")
                            st.write(f"**PBV:** {fmt_num(fund.get('pb'))}x")
                            st.write(f"**PEG:** {fmt_num(fund.get('peg'))}x")
                        with b:
                            st.markdown("#### Kualitas Neraca & Profitabilitas")
                            st.write(f"**ROE:** {pct(fund.get('roe'))}")
                            st.write(f"**ROA:** {pct(fund.get('roa'))}")
                            st.write(f"**NPM:** {pct(fund.get('npm'))}")
                            st.write(f"**DER:** {fmt_num(fund.get('der'))}x")
                            st.write(f"**Current Ratio:** {fmt_num(fund.get('cr'))}x")
                            st.write(f"**Total Debt:** {fmt_num(fund.get('debt'))}")
                            st.write(f"**Equity:** {fmt_num(fund.get('equity'))}")
                            st.write(f"**Assets:** {fmt_num(fund.get('assets'))}")
                            st.write(f"**Website Resolved:** {fund.get('resolved_website') or 'N/A'}")
                            st.write(f"**Website Source:** {fund.get('website_resolution_source') or 'N/A'}")
                            st.write(f"**Gross Margin:** {pct(fund.get('gross_margin'))}")
                            st.write(f"**EBIT Margin:** {pct(fund.get('ebit_margin'))}")
                            st.write(f"**Sector Bucket:** {fund.get('sector_bucket', 'general')}")

                    with tab2:
                        last = tdf.iloc[-1]
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("#### Trader Logic")
                            st.write(f"**RSI (14):** {fmt_num(last['RSI14'])}")
                            st.write(f"**Stochastic K:** {fmt_num(last['StochK'])}")
                            st.write(f"**ATR (14):** {fmt_num(last['ATR14'])}")
                            st.write(f"**Volume Ratio:** {fmt_num(last['Volume_Ratio'])}x")
                            st.write("**Breakout 20D:**", "✅ Ya" if bool(last["Breakout_20D"]) else "❌ Tidak")
                            st.write("**Trend Strong:**", "✅ Ya" if bool(last["Trend_Strong"]) else "❌ Belum")
                            st.write("**Andry Style:**", f"{s_andry:.1f}/100")
                            st.write("**Hengky Style:**", f"{s_hengky:.1f}/100")
                        with c2:
                            st.markdown("#### Value / Growth / Smart Money")
                            st.write("**Bullish Order Block (OB):**", "✅ Terdeteksi" if last["Bullish_OB"] else "❌ Tidak Ada")
                            st.write("**Bullish Fair Value Gap (FVG):**", "✅ Terdeteksi" if last["Bullish_FVG"] else "❌ Tidak Ada")
                            st.write("**Discount Zone:**", "✅ Di Bawah Equilibrium" if last["Discount"] else "❌ Premium Zone")
                            st.write("**Bullish Break of Structure (BOS):**", "✅ Terdeteksi" if last["Bullish_BOS"] else "❌ Tidak Ada")
                            st.write("**Liquidity Sweep:**", "✅ Valid" if last["Liquidity_Sweep"] else "❌ Tidak")
                            st.write("**Bullish ChoCH:**", "✅ Valid" if last["Bullish_ChoCH"] else "❌ Tidak")
                            st.write("**Quality Score:**", f"{s_quality:.1f}/100")
                            st.write("**Growth Score:**", f"{s_growth:.1f}/100")
                            st.write("**Value Score:**", f"{s_value:.1f}/100")
                            st.write("**Trader Score:**", f"{s_trader:.1f}/100")
                            st.write("**SMC Score:**", f"{s_smc:.1f}/100")

                    with tab3:
                        fig = make_subplots(
                            rows=2,
                            cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.05,
                            row_heights=[0.7, 0.3],
                        )
                        fig.add_trace(
                            go.Candlestick(
                                x=tdf.index,
                                open=tdf["Open"],
                                high=tdf["High"],
                                low=tdf["Low"],
                                close=tdf["Close"],
                                name="Candlestick",
                            ),
                            row=1,
                            col=1,
                        )
                        fig.add_trace(go.Scatter(x=tdf.index, y=tdf["SMA20"], mode="lines", name="SMA 20"), row=1, col=1)
                        if "SMA50" in tdf.columns:
                            fig.add_trace(go.Scatter(x=tdf.index, y=tdf["SMA50"], mode="lines", name="SMA 50"), row=1, col=1)
                        if "SMA200" in tdf.columns:
                            fig.add_trace(go.Scatter(x=tdf.index, y=tdf["SMA200"], mode="lines", name="SMA 200"), row=1, col=1)
                        fig.add_trace(go.Bar(x=tdf.index, y=tdf["Volume"], name="Volume"), row=2, col=1)
                        fig.update_layout(
                            xaxis_rangeslider_visible=False,
                            height=650,
                            template="plotly_dark",
                            margin=dict(l=20, r=20, t=20, b=20),
                            legend=dict(orientation="h"),
                        )
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pilih emiten target atau klik tombol 'Eksekusi Analisis Mendalam' di panel sidebar.")
