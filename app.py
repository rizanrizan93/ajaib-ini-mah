import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# SMART ALPHA SCREENER v11.1 - MULTI-TICKER EDITION
# Fundamental + Growth + SMC/ICT + Bandarmology Proxy
# Single-file Streamlit app with Bulk & Deep Dive Mode
# =========================================================

st.set_page_config(page_title="Smart Alpha Screener v11.1", layout="wide")

# -------------------------------
# Helpers
# -------------------------------
def safe_num(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        return float(x)
    except Exception:
        return default


def fmt_num(x, decimals=2, suffix=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    try:
        return f"{x:,.{decimals}f}{suffix}"
    except Exception:
        return "N/A"


def pct(x, decimals=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:.{decimals}f}%"


def get_row_value(df, row_name):
    if df is None or df.empty:
        return np.nan
    try:
        if row_name in df.index:
            s = df.loc[row_name]
            s = s.dropna()
            if len(s) > 0:
                return safe_num(s.iloc[0])
    except Exception:
        pass
    return np.nan


def series_from_stmt(df, row_names):
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    for row_name in row_names:
        if row_name in df.index:
            s = pd.to_numeric(df.loc[row_name], errors="coerce").dropna()
            if len(s) > 0:
                return s
    return pd.Series(dtype="float64")


def cagr(first, last, years):
    if any(pd.isna(v) for v in [first, last]) or first <= 0 or years <= 0:
        return np.nan
    try:
        return (last / first) ** (1 / years) - 1
    except Exception:
        return np.nan


def slope(series):
    s = pd.Series(series).dropna()
    if len(s) < 3:
        return np.nan
    x = np.arange(len(s))
    try:
        return np.polyfit(x, s.values, 1)[0]
    except Exception:
        return np.nan


# -------------------------------
# Data fetch
# -------------------------------
@st.cache_data(ttl=900)
def fetch_price_history(ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, auto_adjust=False, actions=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=str.title)
        df = df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]].copy()
        df = df.dropna(subset=["Close"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker: str) -> dict:
    tk = yf.Ticker(ticker)
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}

    try:
        income_a = tk.income_stmt
    except Exception:
        income_a = pd.DataFrame()
    try:
        balance_a = tk.balance_sheet
    except Exception:
        balance_a = pd.DataFrame()
    try:
        cashflow_a = tk.cashflow
    except Exception:
        cashflow_a = pd.DataFrame()

    market_cap = safe_num(info.get("marketCap"))
    pe = safe_num(info.get("trailingPE"))
    forward_pe = safe_num(info.get("forwardPE"))
    pb = safe_num(info.get("priceToBook"))
    roe = safe_num(info.get("returnOnEquity")) * 100 if info.get("returnOnEquity") is not None else np.nan
    roa = safe_num(info.get("returnOnAssets")) * 100 if info.get("returnOnAssets") is not None else np.nan
    npm = safe_num(info.get("profitMargins")) * 100 if info.get("profitMargins") is not None else np.nan
    opm = safe_num(info.get("operatingMargins")) * 100 if info.get("operatingMargins") is not None else np.nan
    revenue_growth = safe_num(info.get("revenueGrowth")) * 100 if info.get("revenueGrowth") is not None else np.nan
    earnings_growth = safe_num(info.get("earningsGrowth")) * 100 if info.get("earningsGrowth") is not None else np.nan
    peg = safe_num(info.get("pegRatio"))
    beta = safe_num(info.get("beta"))
    dividend_yield = safe_num(info.get("dividendYield")) * 100 if info.get("dividendYield") is not None else np.nan
    fcf = safe_num(info.get("freeCashflow"))
    ocf = safe_num(info.get("operatingCashflow"))
    total_cash = safe_num(info.get("totalCash"))
    total_debt = safe_num(info.get("totalDebt"))
    current_ratio = safe_num(info.get("currentRatio"))
    quick_ratio = safe_num(info.get("quickRatio"))
    debt_to_equity = safe_num(info.get("debtToEquity"))
    if not np.isnan(debt_to_equity) and debt_to_equity > 20:
        debt_to_equity = debt_to_equity / 100.0

    revenue_series = series_from_stmt(income_a, ["Total Revenue", "Operating Revenue", "Revenue"])
    net_income_series = series_from_stmt(income_a, ["Net Income", "Net Income Common Stockholders"])
    gross_profit_series = series_from_stmt(income_a, ["Gross Profit"])
    ocf_series = series_from_stmt(cashflow_a, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex_series = series_from_stmt(cashflow_a, ["Capital Expenditure", "Capital Expenditures"])

    total_assets = get_row_value(balance_a, "Total Assets")
    total_liabilities = get_row_value(balance_a, "Total Liabilities Net Minority Interest")
    total_equity = get_row_value(balance_a, "Stockholders Equity")
    current_assets = get_row_value(balance_a, "Current Assets")
    current_liabilities = get_row_value(balance_a, "Current Liabilities")
    cash_and_equiv = get_row_value(balance_a, "Cash And Cash Equivalents")
    long_term_debt = get_row_value(balance_a, "Long Term Debt")
    short_term_debt = get_row_value(balance_a, "Current Debt")

    revenue_cagr_3y = np.nan
    net_income_cagr_3y = np.nan
    gross_margin_latest = np.nan
    ocf_latest = np.nan
    fcf_latest = np.nan

    if len(revenue_series) >= 2:
        rev = revenue_series.sort_index(ascending=True)
        if len(rev) >= 4:
            revenue_cagr_3y = cagr(rev.iloc[-4], rev.iloc[-1], 3)
        elif len(rev) >= 2:
            revenue_cagr_3y = cagr(rev.iloc[0], rev.iloc[-1], len(rev) - 1)
    if len(net_income_series) >= 2:
        ni = net_income_series.sort_index(ascending=True)
        if len(ni) >= 4:
            net_income_cagr_3y = cagr(ni.iloc[-4], ni.iloc[-1], 3)
        elif len(ni) >= 2:
            net_income_cagr_3y = cagr(ni.iloc[0], ni.iloc[-1], len(ni) - 1)
    if len(gross_profit_series) >= 1 and len(revenue_series) >= 1:
        gp = safe_num(gross_profit_series.sort_index(ascending=True).iloc[-1])
        rev_latest = safe_num(revenue_series.sort_index(ascending=True).iloc[-1])
        if rev_latest and not np.isnan(gp):
            gross_margin_latest = gp / rev_latest * 100
    if len(ocf_series) >= 1:
        ocf_latest = safe_num(ocf_series.sort_index(ascending=True).iloc[-1])
    if len(capex_series) >= 1 and not np.isnan(ocf_latest):
        capex_latest = safe_num(capex_series.sort_index(ascending=True).iloc[-1])
        fcf_latest = ocf_latest + capex_latest

    return {
        "name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": market_cap,
        "pe": pe,
        "forward_pe": forward_pe,
        "pb": pb,
        "peg": peg,
        "roe": roe,
        "roa": roa,
        "npm": npm,
        "opm": opm,
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "revenue_cagr_3y": revenue_cagr_3y * 100 if not np.isnan(revenue_cagr_3y) else np.nan,
        "net_income_cagr_3y": net_income_cagr_3y * 100 if not np.isnan(net_income_cagr_3y) else np.nan,
        "gross_margin_latest": gross_margin_latest,
        "ocf": ocf if not np.isnan(ocf) else ocf_latest,
        "fcf": fcf if not np.isnan(fcf) else fcf_latest,
        "total_cash": total_cash,
        "total_debt": total_debt,
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
        "debt_to_equity": debt_to_equity,
        "beta": beta,
        "dividend_yield": dividend_yield,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "total_equity": total_equity,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "cash_and_equiv": cash_and_equiv,
        "long_term_debt": long_term_debt,
        "short_term_debt": short_term_debt,
    }


# -------------------------------
# Technicals / SMC / ICT
# -------------------------------
def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    box = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + box))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"].fillna(0)).cumsum()


def detect_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    out = df.copy()
    swing_high = pd.Series(False, index=out.index)
    swing_low = pd.Series(False, index=out.index)
    for i in range(left, len(out) - right):
        h = out["High"].iloc[i]
        l = out["Low"].iloc[i]
        if h == out["High"].iloc[i - left:i + right + 1].max():
            swing_high.iloc[i] = True
        if l == out["Low"].iloc[i - left:i + right + 1].min():
            swing_low.iloc[i] = True
    out["Swing_High"] = swing_high
    out["Swing_Low"] = swing_low
    return out


def calculate_technicals(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().dropna().copy()
    if d.empty:
        return d

    d["SMA20"] = d["Close"].rolling(20).mean()
    d["SMA50"] = d["Close"].rolling(50).mean()
    d["SMA200"] = d["Close"].rolling(200).mean()
    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()
    d["RSI14"] = wilder_rsi(d["Close"], 14)
    low14 = d["Low"].rolling(14).min()
    high14 = d["High"].rolling(14).max()
    d["StochK"] = 100 * (d["Close"] - low14) / (high14 - low14)
    d["StochD"] = d["StochK"].rolling(3).mean()
    d["ATR14"] = atr(d, 14)
    d["ATR_PCT"] = d["ATR14"] / d["Close"] * 100
    d["OBV"] = obv(d)
    d["OBV_Slope_10"] = d["OBV"].rolling(10).apply(lambda s: slope(pd.Series(s)), raw=False)
    d["Volume_MA20"] = d["Volume"].rolling(20).mean()
    d["Volume_Z20"] = (d["Volume"] - d["Volume"].rolling(20).mean()) / d["Volume"].rolling(20).std(ddof=0)
    d["Range_20_High"] = d["High"].rolling(20).max()
    d["Range_20_Low"] = d["Low"].rolling(20).min()

    d = detect_swings(d, 3, 3)

    # Inisialisasi list penampung (Jauh lebih cepat & anti-bug tipe Pandas)
    n = len(d)
    bullish_fvg = [False] * n
    bearish_fvg = [False] * n
    bullish_bos = [False] * n
    bearish_bos = [False] * n
    bullish_choch = [False] * n
    bearish_choch = [False] * n
    bullish_sweep = [False] * n
    bearish_sweep = [False] * n

    last_swing_high = np.nan
    last_swing_low = np.nan
    trend_state = 0

    # Ekstraksi array NumPy untuk optimasi pembacaan data di dalam loop
    high_arr = d["High"].values
    low_arr = d["Low"].values
    close_arr = d["Close"].values
    sh_arr = d["Swing_High"].values
    sl_arr = d["Swing_Low"].values

    for i in range(2, n):
        if low_arr[i] > high_arr[i - 2]:
            bullish_fvg[i] = True
        if high_arr[i] < low_arr[i - 2]:
            bearish_fvg[i] = True

        if sh_arr[i - 1]:
            last_swing_high = high_arr[i - 1]
        if sl_arr[i - 1]:
            last_swing_low = low_arr[i - 1]

        close_now = close_arr[i]
        close_prev = close_arr[i - 1]
        high_now = high_arr[i]
        low_now = low_arr[i]

        if not np.isnan(last_swing_high) and close_now > last_swing_high and close_prev <= last_swing_high:
            bullish_bos[i] = True
            if trend_state <= 0:
                bullish_choch[i] = True
            trend_state = 1
        if not np.isnan(last_swing_low) and close_now < last_swing_low and close_prev >= last_swing_low:
            bearish_bos[i] = True
            if trend_state >= 0:
                bearish_choch[i] = True
            trend_state = -1

        if not np.isnan(last_swing_low) and low_now < last_swing_low and close_now > last_swing_low:
            bullish_sweep[i] = True
        if not np.isnan(last_swing_high) and high_now > last_swing_high and close_now < last_swing_high:
            bearish_sweep[i] = True

    # Injeksi masal seluruh matriks data struktur ke DataFrame
    d["Bullish_FVG"] = bullish_fvg
    d["Bearish_FVG"] = bearish_fvg
    d["Bullish_BOS"] = bullish_bos
    d["Bearish_BOS"] = bearish_bos
    d["Bullish_ChoCH"] = bullish_choch
    d["Bearish_ChoCH"] = bearish_choch
    d["Bullish_Sweep"] = bullish_sweep
    d["Bearish_Sweep"] = bearish_sweep

    d["Above_SMA20"] = d["Close"] > d["SMA20"]
    d["Above_SMA50"] = d["Close"] > d["SMA50"]
    d["Above_SMA200"] = d["Close"] > d["SMA200"]
    d["Trend_Stacked"] = (d["SMA20"] > d["SMA50"]) & (d["SMA50"] > d["SMA200"])
    d["Compression_10"] = d["ATR_PCT"].rolling(10).mean() < d["ATR_PCT"].rolling(40).mean()

    recent_low = d["Low"].rolling(20).min()
    recent_high = d["High"].rolling(20).max()
    eq = (recent_high + recent_low) / 2
    d["Premium"] = d["Close"] > eq
    d["Discount"] = d["Close"] < eq
    d["Range_Position"] = (d["Close"] - recent_low) / (recent_high - recent_low)

    return d


# -------------------------------
# Scoring
# -------------------------------
def score_fundamental(f: dict) -> tuple[float, list[str]]:
    score = 0.0
    notes = []

    def add(cond, pts, msg):
        nonlocal score
        if cond:
            score += pts
            notes.append(msg)

    add(not np.isnan(f["roe"]) and f["roe"] >= 15, 12, "ROE kuat")
    add(not np.isnan(f["roa"]) and f["roa"] >= 8, 6, "ROA sehat")
    add(not np.isnan(f["npm"]) and f["npm"] >= 10, 8, "Margin bersih bagus")
    add(not np.isnan(f["revenue_cagr_3y"]) and f["revenue_cagr_3y"] >= 10, 10, "Revenue growth solid")
    add(not np.isnan(f["net_income_cagr_3y"]) and f["net_income_cagr_3y"] >= 10, 8, "Laba tumbuh")
    add(not np.isnan(f["ocf"]) and f["ocf"] > 0, 8, "OCF positif")
    add(not np.isnan(f["fcf"]) and f["fcf"] > 0, 8, "FCF positif")
    add(not np.isnan(f["current_ratio"]) and f["current_ratio"] >= 1.5, 6, "Likuiditas aman")
    add(not np.isnan(f["quick_ratio"]) and f["quick_ratio"] >= 1.0, 4, "Quick ratio aman")
    add(not np.isnan(f["debt_to_equity"]) and f["debt_to_equity"] <= 1.0, 8, "Leverage terkendali")
    add(not np.isnan(f["pe"]) and f["pe"] > 0 and f["pe"] <= 20, 6, "Valuasi masih wajar")
    add(not np.isnan(f["peg"]) and 0 < f["peg"] <= 1.5, 6, "PEG menarik")
    add(not np.isnan(f["pb"]) and f["pb"] <= 3, 4, "PB tidak terlalu mahal")
    return min(score, 100), notes


def score_technical(df: pd.DataFrame) -> tuple[float, list[str]]:
    if df.empty:
        return 0.0, ["Data teknikal kosong"]
    last = df.iloc[-1]
    score = 0.0
    notes = []

    def add(cond, pts, msg):
        nonlocal score
        if cond:
            score += pts
            notes.append(msg)

    add(last["Close"] > last["SMA20"], 8, "Harga di atas SMA20")
    add(last["SMA20"] > last["SMA50"], 8, "Trend jangka pendek naik")
    add(last["SMA50"] > last["SMA200"], 8, "Trend menengah naik")
    add(last["RSI14"] >= 50 and last["RSI14"] <= 70, 8, "RSI sehat")
    add(last["OBV_Slope_10"] > 0, 10, "OBV naik")
    add(last["Volume_Z20"] > 0.5, 6, "Volume mendukung")
    add(bool(last.get("Bullish_BOS", False)), 12, "Bullish BOS")
    add(bool(last.get("Bullish_ChoCH", False)), 8, "Bullish CHOCH")
    add(bool(last.get("Bullish_FVG", False)), 8, "Bullish FVG")
    add(bool(last.get("Bullish_Sweep", False)), 6, "Liquidity sweep re-entry")
    add(bool(last["Discount"]), 8, "Area discount")
    add(bool(last["Compression_10"]), 4, "Compression before expansion")
    return min(score, 100), notes


def final_rating(total_score: float) -> str:
    if total_score >= 85:
        return "A+ / Sangat Layak"
    if total_score >= 75:
        return "A / Layak"
    if total_score >= 65:
        return "B / Watchlist"
    if total_score >= 50:
        return "C / Spekulatif"
    return "D / Hindari"


def process_single_stock(ticker: str, period: str = "1y") -> dict:
    try:
        price_df = fetch_price_history(ticker, period)
        if price_df.empty or len(price_df) < 30:
            return None
        tech_df = calculate_technicals(price_df)
        if tech_df.empty:
            return None
        fundamentals = fetch_fundamentals(ticker)
        if not fundamentals:
            return None
            
        fund_score, _ = score_fundamental(fundamentals)
        tech_score, _ = score_technical(tech_df)
        final_score = round(fund_score * 0.58 + tech_score * 0.42, 1)
        rating = final_rating(final_score)
        last = tech_df.iloc[-1]
        
        return {
            "Ticker": ticker.replace(".JK", ""),
            "Nama": fundamentals.get("name", ticker),
            "Sector": fundamentals.get("sector", "N/A"),
            "Price": last["Close"],
            "Fund Score": round(fund_score, 1),
            "Tech Score": round(tech_score, 1),
            "Final Score": final_score,
            "Status": rating,
            "ROE %": fundamentals.get("roe", np.nan),
            "D/E": fundamentals.get("debt_to_equity", np.nan),
            "Rev Growth %": fundamentals.get("revenue_growth", np.nan),
            "RSI14": last["RSI14"]
        }
    except Exception:
        return None


# -------------------------------
# UI & Layout Mode Selector
# -------------------------------
st.title("📊 Smart Alpha Screener v11.1 - Institutional Edition")
st.caption("Framework Kombinasi Multi-Factor: Value Growth, Momentum & Market Structure (SMC/ICT/Bandarmology Proxy)")

mode = st.sidebar.radio("Pilih Mode Analisis:", ["Bulk Market Screener", "Single Stock Deep Dive"])

if mode == "Bulk Market Screener":
    st.subheader("🔥 Bulk Market Screener - Pemeringkatan Emiten Terbaik (Top 10)")
    st.write("Sistem mendownload data multi-threading, memproses matriks finansial/SMC, lalu merangking dan menampilkan 10 saham terbaik.")
    
    st.sidebar.divider()
    st.sidebar.markdown("### 📝 Input Saham via Dashboard")
    
    custom_input = st.sidebar.text_area(
        "Masukkan Kode Saham (Pisahkan dengan koma):",
        value="BBRI, BBCA, BMRI, BBNI, TLKM, ASII, AMRT, ACES, BRIS, ADRO, PTBA, ANTM, NCKL, SSIA",
        help="Contoh: BBRI, ANTM, SSIA.\nOtomatis membersihkan spasi dan huruf kecil."
    )
    
    period_selection = st.sidebar.selectbox("Periode Histori Harga:", ["6mo", "1y", "2y"], index=1)
    
    raw_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    tickers_to_scan = []
    for t in raw_tickers:
        if t.endswith(".JK") or t.startswith("^"):
            tickers_to_scan.append(t)
        else:
            tickers_to_scan.append(f"{t}.JK")
            
    st.sidebar.info(f"Total saham dalam antrean: **{len(tickers_to_scan)} emiten**")
    if len(tickers_to_scan) > 50:
        st.sidebar.warning("⚠️ Jumlah saham di atas 50 emiten risiko terkena pembatasan (Rate Limit) Yahoo Finance tinggi.")

    if st.button("🚀 Mulai Perhitungan Seluruh Pasar"):
        if not tickers_to_scan:
            st.error("Antrean saham kosong. Silakan masukkan kode emiten terlebih dahulu di sidebar.")
            st.stop()
            
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(process_single_stock, t, period_selection): t for t in tickers_to_scan}
            completed = 0
            for future in as_completed(future_to_ticker):
                completed += 1
                ticker = future_to_ticker[future]
                status_text.text(f"Memproses {completed}/{len(tickers_to_scan)}: {ticker}")
                progress_bar.progress(completed / len(tickers_to_scan))
                
                res = future.result()
                if res is not None:
                    results.append(res)
                    
        status_text.success("✅ Pemrosesan Selesai!")
        
        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values(by="Final Score", ascending=False).reset_index(drop=True)
            
            df_top10 = df_results.head(10)
            
            st.markdown("### 🏆 Top 10 Saham Terbaik Hasil Perhitungan")
            st.dataframe(
                df_top10, 
                use_container_width=True, 
                hide_index=False,
                column_config={
                    "Final Score": st.column_config.NumberColumn("Final Score", format="%.1f 🔥"),
                    "Price": st.column_config.NumberColumn("Last Price", format="Rp %,.0f"),
                    "Fund Score": st.column_config.NumberColumn("Fund Score"),
                    "Tech Score": st.column_config.NumberColumn("Tech Score"),
                    "ROE %": st.column_config.NumberColumn("ROE", format="%.1f%%"),
                    "Rev Growth %": st.column_config.NumberColumn("Rev Growth", format="%.1f%%"),
                    "D/E": st.column_config.NumberColumn("D/E", format="%.2f"),
                    "RSI14": st.column_config.NumberColumn("RSI (14)", format="%.1f"),
                }
            )
            
            if len(df_results) > 10:
                with st.expander("👀 Lihat Sisa Saham Hasil Evaluasi Lainnya"):
                    st.dataframe(df_results.iloc[10:], use_container_width=True, hide_index=False)
                    
            layak_count = len(df_results[df_results["Final Score"] >= 75])
            st.info(f"Dari total {len(results)} saham teranalisis, terdapat **{layak_count} emiten** yang masuk kualifikasi institusional grade (Score >= 75).")
        else:
            st.error("Gagal memproses data. Kemungkinan besar IP terkena Rate Limit sementara oleh Yahoo Finance.")

else:
    # =========================================================
    # SINGLE STOCK DEEP DIVE MODE
    # =========================================================
    st.sidebar.header("⚙️ Deep Dive Settings")
    ticker_input = st.sidebar.text_input("Ticker", "BBRI.JK")
    benchmark_input = st.sidebar.text_input("Benchmark", "^JKSE")
    period = st.sidebar.selectbox("Periode", ["6mo", "1y", "2y", "5y"], index=1)
    show_benchmark = st.sidebar.checkbox("Tampilkan benchmark", value=True)
    st.sidebar.divider()
    st.sidebar.subheader("Filter prinsip")
    min_roe = st.sidebar.slider("Min ROE %", 0, 40, 15)
    max_de = st.sidebar.slider("Max Debt/Equity", 0.0, 5.0, 1.0, 0.1)
    min_rev_growth = st.sidebar.slider("Min Revenue Growth %", -20, 50, 10)

    if ticker_input.strip():
        formatted_ticker = ticker_input.strip().upper()
        if not formatted_ticker.endswith(".JK") and not formatted_ticker.startswith("^"):
            formatted_ticker = f"{formatted_ticker}.JK"
            
        with st.spinner(f"Mengambil data mendalam {formatted_ticker}..."):
            price_df = fetch_price_history(formatted_ticker, period)
            fundamentals = fetch_fundamentals(formatted_ticker)

        if price_df.empty:
            st.error("Data harga tidak tersedia. Cek format ticker.")
            st.stop()

        tech_df = calculate_technicals(price_df)
        if tech_df.empty or len(tech_df) < 30:
            st.error("Data historis terlalu pendek untuk analisis yang valid.")
            st.stop()

        fund_score, fund_notes = score_fundamental(fundamentals)
        tech_score, tech_notes = score_technical(tech_df)

        final_score = round(fund_score * 0.58 + tech_score * 0.42, 1)
        rating = final_rating(final_score)

        filter_ok = (
            (np.isnan(fundamentals["roe"]) or fundamentals["roe"] >= min_roe)
            and (np.isnan(fundamentals["debt_to_equity"]) or fundamentals["debt_to_equity"] <= max_de)
            and (np.isnan(fundamentals["revenue_growth"]) or fundamentals["revenue_growth"] >= min_rev_growth)
        )

        last = tech_df.iloc[-1]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Final Score", f"{final_score}/100", rating)
        c2.metric("Fundamental", f"{fund_score:.1f}/100")
        c3.metric("Technical", f"{tech_score:.1f}/100")
        c4.metric("Last Close", fmt_num(last["Close"], 2))
        c5.metric("Trend", "Bullish" if last["Close"] > last["SMA50"] else "Neutral/Bearish")

        bullish_structure = bool(last["Bullish_BOS"]) or bool(last["Bullish_ChoCH"]) or bool(last["Bullish_FVG"]) or bool(last["Bullish_Sweep"])
        quality_ok = (not np.isnan(fundamentals["roe"]) and fundamentals["roe"] >= min_roe) and (not np.isnan(fundamentals["debt_to_equity"]) and fundamentals["debt_to_equity"] <= max_de)

        if final_score >= 75 and quality_ok and bullish_structure:
            st.success("LAYAK AKUMULASI BERTAHAP: kualitas fundamental dan struktur harga mendukung.")
        elif final_score >= 65:
            st.warning("WATCHLIST: kualitas cukup, tetapi konfirmasi momentum/struktur belum ideal.")
        else:
            st.error("HINDARI / TUNGGU: risiko masih dominan.")

        fund_tab, tech_tab, chart_tab, logic_tab = st.tabs(["Fundamental", "SMC / ICT", "Chart", "Scoring"])

        with fund_tab:
            st.subheader("Fundamental & Solvabilitas")
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("ROE", pct(fundamentals["roe"]))
            a2.metric("ROA", pct(fundamentals["roa"]))
            a3.metric("NPM", pct(fundamentals["npm"]))
            a4.metric("OPM", pct(fundamentals["opm"]))

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Revenue CAGR 3Y", pct(fundamentals["revenue_cagr_3y"]))
            b2.metric("Net Income CAGR 3Y", pct(fundamentals["net_income_cagr_3y"]))
            b3.metric("OCF", fmt_num(fundamentals["ocf"], 0))
            b4.metric("FCF", fmt_num(fundamentals["fcf"], 0))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PER", fmt_num(fundamentals["pe"]))
            c2.metric("PEG", fmt_num(fundamentals["peg"]))
            c3.metric("PBV", fmt_num(fundamentals["pb"]))
            c4.metric("D/E", fmt_num(fundamentals["debt_to_equity"]))

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Current Ratio", fmt_num(fundamentals["current_ratio"]))
            d2.metric("Quick Ratio", fmt_num(fundamentals["quick_ratio"]))
            d3.metric("Total Cash", fmt_num(fundamentals["total_cash"], 0))
            d4.metric("Total Debt", fmt_num(fundamentals["total_debt"], 0))

            st.markdown("**Kriteria fundamental yang terpenuhi:**")
            st.write("- " + ("\n- ".join(fund_notes) if fund_notes else "Tidak ada sinyal kuat"))

        with tech_tab:
            st.subheader("SMC / ICT / Bandarmology Proxy")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("RSI14", fmt_num(last["RSI14"]))
            t2.metric("Stoch K", fmt_num(last["StochK"]))
            t3.metric("ATR %", fmt_num(last["ATR_PCT"]))
            t4.metric("OBV Slope 10", fmt_num(last["OBV_Slope_10"]))

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Bullish BOS", "Yes" if bool(last["Bullish_BOS"]) else "No")
            e2.metric("Bullish CHOCH", "Yes" if bool(last["Bullish_ChoCH"]) else "No")
            e3.metric("Bullish FVG", "Yes" if bool(last["Bullish_FVG"]) else "No")
            e4.metric("Bullish Sweep", "Yes" if bool(last["Bullish_Sweep"]) else "No")

            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Above SMA20", "Yes" if bool(last["Above_SMA20"]) else "No")
            f2.metric("Above SMA50", "Yes" if bool(last["Above_SMA50"]) else "No")
            f3.metric("Trend Stacked", "Yes" if bool(last["Trend_Stacked"]) else "No")
            f4.metric("Discount Zone", "Yes" if bool(last["Discount"]) else "No")

            st.markdown("**Interpretasi teknikal:**")
            st.write("- " + ("\n- ".join(tech_notes) if tech_notes else "Tidak ada sinyal kuat"))

        with chart_tab:
            st.subheader("Harga, Trend, dan Volume")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=tech_df.index, open=tech_df["Open"], high=tech_df["High"], low=tech_df["Low"], close=tech_df["Close"], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["SMA20"], name="SMA20", mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["SMA50"], name="SMA50", mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["SMA200"], name="SMA200", mode="lines"), row=1, col=1)

            recent = tech_df.tail(60)
            bos_idx = recent.index[recent["Bullish_BOS"]]
            fvg_idx = recent.index[recent["Bullish_FVG"]]
            sweep_idx = recent.index[recent["Bullish_Sweep"]]
            if len(bos_idx) > 0:
                fig.add_trace(go.Scatter(x=bos_idx, y=tech_df.loc[bos_idx, "Close"], mode="markers", name="BOS", marker=dict(size=10, symbol="triangle-up")), row=1, col=1)
            if len(fvg_idx) > 0:
                fig.add_trace(go.Scatter(x=fvg_idx, y=tech_df.loc[fvg_idx, "Low"], mode="markers", name="FVG", marker=dict(size=9, symbol="diamond")), row=1, col=1)
            if len(sweep_idx) > 0:
                fig.add_trace(go.Scatter(x=sweep_idx, y=tech_df.loc[sweep_idx, "Low"], mode="markers", name="Sweep", marker=dict(size=9, symbol="x")), row=1, col=1)

            fig.add_trace(go.Bar(x=tech_df.index, y=tech_df["Volume"], name="Volume"), row=2, col=1)
            fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df["Volume_MA20"], name="Vol MA20", mode="lines"), row=2, col=1)

            if show_benchmark:
                try:
                    bm = fetch_price_history(benchmark_input.strip(), period)
                    if not bm.empty:
                        bm_norm = bm["Close"] / bm["Close"].iloc[0] * tech_df["Close"].iloc[0]
                        fig.add_trace(go.Scatter(x=bm_norm.index, y=bm_norm, name=f"Benchmark {benchmark_input}", mode="lines"), row=1, col=1)
                except Exception:
                    pass

            fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False, legend=dict(orientation="h"), margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with logic_tab:
            st.subheader("Scoring Engine")
            l1, l2 = st.columns(2)
            with l1:
                st.metric("Fundamental Score", f"{fund_score:.1f}/100")
                st.metric("Technical Score", f"{tech_score:.1f}/100")
                st.metric("Final Score", f"{final_score}/100")
                st.metric("Rating", rating)
            with l2:
                st.write("**Filter status**")
                st.write(f"- ROE >= {min_roe}%: {'Pass' if np.isnan(fundamentals['roe']) or fundamentals['roe'] >= min_roe else 'Fail'}")
                st.write(f"- D/E <= {max_de}: {'Pass' if np.isnan(fundamentals['debt_to_equity']) or fundamentals['debt_to_equity'] <= max_de else 'Fail'}")
                st.write(f"- Revenue Growth >= {min_rev_growth}%: {'Pass' if np.isnan(fundamentals['revenue_growth']) or fundamentals['revenue_growth'] >= min_rev_growth else 'Fail'}")
                st.write(f"- Price structure bullish: {'Pass' if bullish_structure else 'Fail'}")

            st.markdown("### Data ringkas")
            summary = pd.DataFrame([
                ["Nama", fundamentals["name"]],
                ["Sector", fundamentals["sector"]],
                ["Industry", fundamentals["industry"]],
                ["Close", last["Close"]],
                ["SMA20", last["SMA20"]],
                ["SMA50", last["SMA50"]],
                ["SMA200", last["SMA200"]],
                ["RSI14", last["RSI14"]],
                ["ATR%", last["ATR_PCT"]],
                ["Revenue Growth %", fundamentals["revenue_growth"]],
                ["Revenue CAGR 3Y %", fundamentals["revenue_cagr_3y"]],
                ["ROE %", fundamentals["roe"]],
                ["ROA %", fundamentals["roa"]],
                ["NPM %", fundamentals["npm"]],
                ["D/E", fundamentals["debt_to_equity"]],
                ["Current Ratio", fundamentals["current_ratio"]],
                ["PEG", fundamentals["peg"]],
            ], columns=["Metric", "Value"])
            st.dataframe(summary, use_container_width=True, hide_index=True)

st.caption("Catatan: beberapa metrik fundamental dari sumber publik dapat tidak lengkap atau berbeda antar emiten / bursa. Selalu gunakan sebagai screening, bukan satu-satunya dasar keputusan.")