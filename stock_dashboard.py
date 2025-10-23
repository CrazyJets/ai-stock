import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from datetime import date, datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup
import warnings
import time
import re
from io import BytesIO
from urllib.parse import urljoin, urlparse
import importlib

# PDF extraction
# Try several PDF libraries gracefully. Streamlit environments sometimes don't have PyPDF2 installed.
PdfReader = None
USE_PYMUPDF = False
try:
    from PyPDF2 import PdfReader  # preferred
except Exception:
    try:
        import fitz  # PyMuPDF as a fallback
        USE_PYMUPDF = True
    except Exception:
        # Neither PyPDF2 nor PyMuPDF available. We'll handle this gracefully in extraction functions.
        PdfReader = None
        USE_PYMUPDF = False

# cloudscraper is helpful to handle common Cloudflare checks in a non-invasive manner.
# If you don't have it in your environment, we fall back to a simple requests-based scraper.
# To enable the Cloudflare-handling scraper, install: pip install cloudscraper
HAS_CLOUDSCRAPER = False
SCRAPER = None
try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
    # create_scraper may raise in odd environments; guard it
    try:
        SCRAPER = cloudscraper.create_scraper()
    except Exception:
        SCRAPER = None
        HAS_CLOUDSCRAPER = False
except Exception:
    cloudscraper = None
    HAS_CLOUDSCRAPER = False
    SCRAPER = None

# Fallback simple scraper wrapper to provide .get() when cloudscraper isn't available.
class _RequestsScraper:
    def get(self, url, timeout=15, headers=None, **kwargs):
        # Mirror requests.Response interface for our callers
        return requests.get(url, timeout=timeout, headers=headers or {}, **kwargs)

# Ensure SCRAPER is set to something with a .get method
if SCRAPER is None:
    SCRAPER = _RequestsScraper()

warnings.filterwarnings('ignore')

st.set_page_config(page_title="💹 AI + CSE Stock Dashboard", layout="wide")

# ---------------- Helper Functions -----------------
def normalize_cse_ticker_for_yf(ticker: str) -> str:
    """Return Yahoo-format CSE ticker (e.g., SAMP.N0000 -> SAMP.N0000.CM)"""
    t = ticker.upper().strip()
    t = t.replace('-', '.')
    t = t.replace(' ', '.')
    if t.endswith('.CM'):
        t = t[:-3]
    return t + ".CM"

def normalize_cse_ticker_for_cselk(ticker: str) -> str:
    """Return CSE package-style ticker (SAMP.N0000), remove any .CM, replace - with ."""
    t = ticker.upper().strip()
    t = t.replace('-', '.')
    t = t.replace(' ', '.')
    if t.endswith('.CM'):
        t = t[:-3]
    return t

def normalize_general_ticker(ticker: str) -> str:
    """General upper-case cleanup for non-CSE/Yahoo input."""
    return ticker.upper().strip()

@st.cache_resource
def get_ticker_resource(tkr: str):
    return yf.Ticker(tkr)

@st.cache_data
def get_yf_data(tkr: str, start: date, end: date):
    try:
        tk = get_ticker_resource(tkr)
        hist = tk.history(start=start, end=end)
        hist = pd.DataFrame(hist).copy()
    except Exception:
        return pd.DataFrame(), {}
    try:
        info = dict(tk.info or {})
    except Exception:
        info = {}
    return hist, info

# Indicator calculation
def calculate_indicators(df):
    # ensure required column
    if 'Close' not in df.columns:
        return df
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    delta = df['Close'].diff()
    gain = delta.where(delta>0, 0).rolling(14).mean()
    loss = -delta.where(delta<0, 0).rolling(14).mean()
    rs = gain/loss
    df['RSI'] = 100 - (100/(1+rs))
    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_upper'] = bb_mid + (bb_std*2)
    df['BB_lower'] = bb_mid - (bb_std*2)
    df['Vol_SMA'] = df['Volume'].rolling(20).mean() if 'Volume' in df.columns else np.nan
    df['Volatility'] = df['Close'].pct_change().rolling(10).std()*np.sqrt(252)*100
    return df

def buy_sell_signal(df):
    if df.empty or 'MACD' not in df.columns or 'MACD_signal' not in df.columns or 'RSI' not in df.columns:
        return "HOLD", "Insufficient data"
    latest_macd = df['MACD'].iloc[-1]
    latest_signal = df['MACD_signal'].iloc[-1]
    latest_rsi = df['RSI'].iloc[-1]
    if latest_macd > latest_signal and latest_rsi < 70:
        return "BUY", f"MACD crossover & RSI {latest_rsi:.2f}"
    elif latest_macd < latest_signal and latest_rsi > 30:
        return "SELL", f"MACD down & RSI {latest_rsi:.2f}"
    else:
        return "HOLD", f"No strong signal, RSI {latest_rsi:.2f}"

def ai_market_analysis(df, info, signal, reason):
    if df.empty or len(df) < 2:
        return ["No analysis available (insufficient data)"]
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    pc = (latest['Close'] - prev['Close']) / prev['Close'] * 100
    vol_ratio = latest['Volume'] / latest.get('Vol_SMA', 1) if latest.get('Vol_SMA', None) is not None else 1
    insights = [f"📊 Change: {pc:+.2f}% → {signal} — {reason}"]
    if latest.get('RSI', 50) > 70: insights.append(f"⚠️ RSI {latest['RSI']:.1f} Overbought risk")
    elif latest.get('RSI', 50) < 30: insights.append(f"💡 RSI {latest['RSI']:.1f} Oversold opportunity")
    if latest.get('MACD', 0) > latest.get('MACD_signal', 0): insights.append("📈 MACD bullish momentum")
    else: insights.append("📉 MACD bearish momentum")
    if vol_ratio > 1.5: insights.append("🔥 High volume confirms strong move")
    elif vol_ratio < 0.7: insights.append("📊 Weak volume")
    return insights

# Technical Strength Meter
def calculate_strength(df):
    if df.empty:
        return 50
    latest = df.iloc[-1]
    score = 50
    if latest.get('RSI', 50) > 70: score -= 20
    elif latest.get('RSI', 50) < 30: score += 20
    if latest.get('MACD', 0) > latest.get('MACD_signal', 0): score += 15
    else: score -= 15
    # Bollinger position
    if latest.get('Close', 0) > latest.get('BB_upper', np.inf): score -= 10
    elif latest.get('Close', 0) < latest.get('BB_lower', -np.inf): score += 10
    # Volume
    vol_ratio = latest.get('Volume', 1) / latest.get('Vol_SMA', 1) if latest.get('Vol_SMA', None) is not None else 1
    if vol_ratio > 1.5: score += 5
    elif vol_ratio < 0.7: score -= 5
    return max(0, min(100, score))

# ML Model
class StockML:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    def prepare_data(self, df):
        df = df.dropna()
        X = df[['Close','SMA_20','SMA_50','RSI','MACD','MACD_signal']]
        y = df['Close'].shift(-1).dropna()
        X = X.iloc[:-1]; y = y.iloc[:-1]
        return X, y
    def train(self, df):
        X, y = self.prepare_data(df)
        if len(X) < 30: return None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(self.scaler.fit_transform(X_train), y_train)
        score = self.model.score(self.scaler.transform(X_test), y_test)
        return score, X.iloc[-1:]
    def predict_next(self, features):
        return self.model.predict(self.scaler.transform(features))[0]

# ---------------- Network helpers with Cloudflare handling ----------------
def is_blocked_by_cf(text, status_code):
    if status_code in (429, 503):
        return True
    block_signals = [
        "cf-browser-verification",
        "cloudflare",
        "Checking your browser",
        "Please enable JavaScript",
        "bot verification",
        "are you human"
    ]
    lower = (text or "").lower()
    return any(s in lower for s in block_signals)

def get_html(url, use_scraper=True, timeout=15):
    """Return (text, final_url, status_code) or (None, url, status) if blocked/fails."""
    headers = {"User-Agent": "ai-stock-dashboard/1.0 (+https://github.com/CrazyJets)"}
    try:
        if use_scraper and HAS_CLOUDSCRAPER:
            r = SCRAPER.get(url, timeout=timeout, headers=headers)
        else:
            # Use SCRAPER (requests-based fallback) or plain requests
            r = SCRAPER.get(url, timeout=timeout, headers=headers)
        text = getattr(r, "text", None) or ""
        final_url = getattr(r, "url", url)
        status_code = getattr(r, "status_code", None)
        if is_blocked_by_cf(text, status_code):
            return None, final_url, status_code
        return text, final_url, status_code
    except Exception:
        # last-resort: try plain requests
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            text = r.text or ""
            if is_blocked_by_cf(text, r.status_code):
                return None, r.url, r.status_code
            return text, r.url, r.status_code
        except Exception:
            return None, url, None

# News & generic scrapers
def scrape_site(url):
    try:
        text, final_url, status = get_html(url)
        if text is None:
            return []  # blocked or unavailable
        soup = BeautifulSoup(text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            text_link = a.text.strip()
            link = urljoin(final_url, a['href'])
            if text_link and (link.lower().endswith(".pdf") or re.search(r"\.pdf(\?|$)", link.lower()) or 'pdf' in link.lower()):
                links.append((text_link, link))
        return links
    except Exception:
        return []

def fetch_yahoo_news(symbol):
    try:
        t = yf.Ticker(symbol)
        return t.news
    except Exception:
        return []

# ---------------- cse.lk integration (optional) ----------------
CSE_MODULE = None
CSE_AVAILABLE = False
# Try to import cse.lk (the user requested package). Many packages expose a submodule or top-level name;
# try a few import patterns safely.
try:
    CSE_MODULE = importlib.import_module("cse.lk")
    CSE_AVAILABLE = True
except Exception:
    try:
        CSE_MODULE = importlib.import_module("cse")
        CSE_AVAILABLE = True
    except Exception:
        try:
            CSE_MODULE = importlib.import_module("cse_lk")
            CSE_AVAILABLE = True
        except Exception:
            CSE_MODULE = None
            CSE_AVAILABLE = False

def get_cse_company_details(cse_symbol: str):
    """Wrapper to get company details from cse.lk package if available.
    This function tries common call patterns and returns either a dict/object
    or a clear error dict if the package isn't installed or API differs.
    """
    if not CSE_AVAILABLE or CSE_MODULE is None:
        return {"error": "cse.lk package not installed. Install with: pip install cse.lk"}
    # Try several possible exposed APIs (best-effort)
    try:
        # If module exposes a 'get_company' function
        if hasattr(CSE_MODULE, "get_company"):
            return CSE_MODULE.get_company(cse_symbol)
        if hasattr(CSE_MODULE, "get_stock"):
            return CSE_MODULE.get_stock(cse_symbol)
        # Some libraries may expose a Client/Company object
        if hasattr(CSE_MODULE, "Company"):
            try:
                return CSE_MODULE.Company(cse_symbol)
            except Exception:
                # try constructing
                try:
                    return CSE_MODULE.Company(symbol=cse_symbol)
                except Exception:
                    pass
        # try a generic fetch function
        if hasattr(CSE_MODULE, "fetch"):
            return CSE_MODULE.fetch(cse_symbol)
        # Last resort: return module dir as info to help user debug
        return {"error": "cse.lk module loaded but API not recognized. Available attributes: " + ", ".join(dir(CSE_MODULE)[:20])}
    except Exception as e:
        return {"error": f"cse.lk call failed: {e}"}

def _to_df_from_records(records, date_key_candidates=("date", "Date", "timestamp", "datetime")):
    """Helper to convert list-of-dicts records to a DataFrame with Date index and OHLCV columns if possible."""
    if not records:
        return pd.DataFrame()
    try:
        df = pd.DataFrame(records)
        # identify date column
        date_col = None
        for c in date_key_candidates:
            if c in df.columns:
                date_col = c
                break
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
        # normalize column names to standard OHLCV names
        colmap = {}
        for c in df.columns:
            cl = c.lower()
            if "open" in cl and "open" not in colmap:
                colmap[c] = "Open"
            elif "high" in cl and "high" not in colmap:
                colmap[c] = "High"
            elif "low" in cl and "low" not in colmap:
                colmap[c] = "Low"
            elif ("close" in cl or "price" in cl) and "Close" not in colmap:
                colmap[c] = "Close"
            elif "volume" in cl and "Volume" not in colmap:
                colmap[c] = "Volume"
        df = df.rename(columns=colmap)
        # ensure columns exist
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = np.nan
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        return df
    except Exception:
        return pd.DataFrame()

def get_cse_history(cse_symbol: str, start_dt: date, end_dt: date):
    """Try multiple approaches to retrieve historical price data from the cse.lk package.
    Returns (df, meta) where df is DataFrame or empty DataFrame on failure, meta is dict with debug info.
    """
    meta = {"attempts": [], "used": None}
    if not CSE_AVAILABLE or CSE_MODULE is None:
        meta["attempts"].append("cse.lk not installed")
        return pd.DataFrame(), meta

    # Convert dates to strings if needed
    start_str = start_dt.isoformat()
    end_str = end_dt.isoformat()

    # Candidate function names / patterns to try
    candidates = [
        ("historical_prices", lambda mod: getattr(mod, "historical_prices", None)),
        ("get_history", lambda mod: getattr(mod, "get_history", None)),
        ("get_stock_history", lambda mod: getattr(mod, "get_stock_history", None)),
        ("historical", lambda mod: getattr(mod, "historical", None)),
        ("prices", lambda mod: getattr(mod, "prices", None)),
        ("fetch_prices", lambda mod: getattr(mod, "fetch_prices", None)),
    ]
    # Also try if module has a Client or API class that can be constructed
    try:
        # If module exposes a function that accepts (symbol, start, end)
        for name, getter in candidates:
            fn = getter(CSE_MODULE)
            if fn is None:
                meta["attempts"].append(f"{name}: not found")
                continue
            meta["attempts"].append(f"{name}: found, trying call")
            try:
                # Try typical signatures
                try:
                    res = fn(cse_symbol, start_str, end_str)
                except TypeError:
                    try:
                        res = fn(symbol=cse_symbol, start=start_str, end=end_str)
                    except TypeError:
                        try:
                            res = fn(cse_symbol)
                        except Exception as e:
                            raise
                # Normalize result
                if isinstance(res, pd.DataFrame):
                    meta["used"] = name
                    df = res.copy()
                    # if index isn't datetime, try to find a date column
                    if not pd.api.types.is_datetime64_any_dtype(df.index):
                        df = _to_df_from_records(df.reset_index().to_dict(orient="records"))
                    else:
                        # try to standardize column names
                        cols_lower = {c.lower(): c for c in df.columns}
                        rename = {}
                        for k in cols_lower:
                            if "close" in k and "Close" not in df.columns:
                                rename[cols_lower[k]] = "Close"
                            if "open" in k and "Open" not in df.columns:
                                rename[cols_lower[k]] = "Open"
                            if "high" in k and "High" not in df.columns:
                                rename[cols_lower[k]] = "High"
                            if "low" in k and "Low" not in df.columns:
                                rename[cols_lower[k]] = "Low"
                            if "volume" in k and "Volume" not in df.columns:
                                rename[cols_lower[k]] = "Volume"
                        if rename:
                            df = df.rename(columns=rename)
                        # ensure OHLCV presence
                        for col in ["Open", "High", "Low", "Close", "Volume"]:
                            if col not in df.columns:
                                df[col] = np.nan
                        df = df[["Open", "High", "Low", "Close", "Volume"]]
                    return df.sort_index(), meta
                # If it's list-of-dicts
                if isinstance(res, (list, tuple)):
                    df = _to_df_from_records(list(res))
                    if not df.empty:
                        meta["used"] = name
                        return df.sort_index(), meta
                # If it's an object with .to_dataframe or .to_df
                if hasattr(res, "to_dataframe"):
                    try:
                        df = res.to_dataframe()
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            meta["used"] = name
                            return df.sort_index(), meta
                    except Exception:
                        pass
                if hasattr(res, "to_df"):
                    try:
                        df = res.to_df()
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            meta["used"] = name
                            return df.sort_index(), meta
                    except Exception:
                        pass
                # If returns dict with 'history' key
                if isinstance(res, dict) and "history" in res:
                    df = _to_df_from_records(res["history"])
                    if not df.empty:
                        meta["used"] = name
                        return df.sort_index(), meta
                # fallback: attempt to parse repr for csv-like data (unlikely)
                meta["attempts"].append(f"{name}: unrecognized return type {type(res)}")
            except Exception as e:
                meta["attempts"].append(f"{name}: call failed: {e}")
        # Try client pattern
        if hasattr(CSE_MODULE, "Client") or hasattr(CSE_MODULE, "CSEClient") or hasattr(CSE_MODULE, "API"):
            for cls_name in ("Client", "CSEClient", "API"):
                cls = getattr(CSE_MODULE, cls_name, None)
                if cls is None:
                    meta["attempts"].append(f"{cls_name}: not found")
                    continue
                meta["attempts"].append(f"{cls_name}: found, attempting instantiation")
                try:
                    client = cls()
                    if hasattr(client, "historical_prices"):
                        try:
                            res = client.historical_prices(cse_symbol, start_str, end_str)
                            df = res if isinstance(res, pd.DataFrame) else _to_df_from_records(res)
                            if not df.empty:
                                meta["used"] = f"{cls_name}.historical_prices"
                                return df.sort_index(), meta
                        except Exception as e:
                            meta["attempts"].append(f"{cls_name}.historical_prices call failed: {e}")
                    # try other client methods
                    for method in ("get_history", "get_stock_history", "historical"):
                        m = getattr(client, method, None)
                        if m is None:
                            meta["attempts"].append(f"{cls_name}.{method}: not found")
                            continue
                        try:
                            res = m(cse_symbol, start_str, end_str)
                            df = res if isinstance(res, pd.DataFrame) else _to_df_from_records(res)
                            if not df.empty:
                                meta["used"] = f"{cls_name}.{method}"
                                return df.sort_index(), meta
                        except Exception as e:
                            meta["attempts"].append(f"{cls_name}.{method} call failed: {e}")
                except Exception as e:
                    meta["attempts"].append(f"{cls_name}: instantiation failed: {e}")
    except Exception as e:
        meta["attempts"].append(f"unexpected error: {e}")

    # Nothing worked
    meta["attempts"].append("no recognized history API returned data")
    return pd.DataFrame(), meta

# ---------------- Sidebar controls ----------------
st.sidebar.title("📊 Dashboard Controls")
# use key so we can modify via session_state when normalizing
ticker_input = st.sidebar.text_input("Ticker", "WIND-N0000.CM", key="ticker_input")
platform = st.sidebar.selectbox("Choose platform", ["Yahoo Finance", "CSE (Colombo Stock Exchange)"])
use_cse_checkbox = (platform.startswith("CSE"))
start_date = st.sidebar.date_input("Start Date", date(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())
show_prediction = st.sidebar.checkbox("🔮 Show ML Prediction", value=True)
enable_news = st.sidebar.checkbox("📰 Show Market News", value=True)
news_source = st.sidebar.selectbox("Choose News Source", ["Yahoo Finance", "EconomyNext business", "EconomyNext market"])

# Add explicit normalization controls for users who want the old button behavior
st.sidebar.markdown("### CSE ticker helpers")
auto_map_cm = st.sidebar.checkbox("Auto-map .CM for CSE tickers", value=True)
if st.sidebar.button("Normalize ticker for selected platform"):
    if platform.startswith("CSE"):
        normalized = normalize_cse_ticker_for_cselk(ticker_input) if auto_map_cm else normalize_general_ticker(ticker_input)
        st.session_state["ticker_input"] = normalized
        st.experimental_rerun()
    else:
        st.session_state["ticker_input"] = normalize_general_ticker(ticker_input)
        st.experimental_rerun()

# Responsive small CSS tweaks for mobile
st.markdown("""
<style>
/* Make header and content scale better on mobile */
@media (max-width: 600px) {
  .css-18e3th9 { padding: 8px 12px; } /* main content padding */
  .stButton>button { padding: .45rem .8rem; }
  .stTextInput>div>div>input { font-size: 14px; }
}
</style>
""", unsafe_allow_html=True)

# Inform user on availability of PDF extraction libraries (non-blocking)
if not (PdfReader is not None or USE_PYMUPDF):
    st.sidebar.info("PDF extraction libraries not found. PDF text extraction will be disabled. To enable, install PyPDF2 or PyMuPDF in your environment.")

# Inform user about cloudscraper availability
if not HAS_CLOUDSCRAPER:
    st.sidebar.info(
        "Cloudscraper not available. Requests-based fallback will be used; some Cloudflare-protected sites may block access. "
        "To enable better scraping, install cloudscraper: pip install cloudscraper"
    )

# Inform user about cse.lk status
if not CSE_AVAILABLE:
    st.sidebar.info("cse.lk not found. To enable CSE price/company data, install: pip install cse.lk")

# Load data
if (upload_data := st.sidebar.file_uploader("📤 Upload CSV/Excel", type=["csv", "xlsx"])) is not None:
    if upload_data.name.endswith(".csv"):
        hist = pd.read_csv(upload_data, parse_dates=[0])
    else:
        hist = pd.read_excel(upload_data, parse_dates=[0])
    hist = hist.set_index(hist.columns[0])
    info = {}
else:
    # Normalize ticker according to selected platform and try to fetch data
    if platform.startswith("CSE"):
        # For the UI we accept SAMP-N0000 or SAMP.N0000 or SAMP N0000. Use a clean CSE form for cse.lk package.
        cse_symbol = normalize_cse_ticker_for_cselk(ticker_input)
        yf_symbol = normalize_cse_ticker_for_yf(ticker_input) if auto_map_cm else normalize_general_ticker(ticker_input)

        # First: attempt to fetch history using cse.lk
        st.sidebar.write("Attempting to load historical prices from cse.lk package (if installed)...")
        hist_cse, meta = get_cse_history(cse_symbol, start_date, end_date)
        if not hist_cse.empty:
            # Ensure index and numeric types
            try:
                hist = hist_cse.copy()
                # if index is not datetime, try to convert
                if not pd.api.types.is_datetime64_any_dtype(hist.index):
                    hist.index = pd.to_datetime(hist.index)
                hist = hist.sort_index()
                # ensure column types numeric
                for col in ['Open','High','Low','Close','Volume']:
                    if col in hist.columns:
                        hist[col] = pd.to_numeric(hist[col], errors='coerce')
                info = {"source": "cse.lk", "cse_meta": meta}
                st.sidebar.success(f"Loaded {len(hist)} rows from cse.lk (method: {meta.get('used')}).")
            except Exception as e:
                st.sidebar.error(f"Failed to normalize cse.lk data: {e}")
                hist = pd.DataFrame()
                info = {}
        else:
            # Fallback: use Yahoo - try several variants to be resilient
            st.sidebar.warning("cse.lk history not available or failed. Falling back to Yahoo Finance for price history.")
            hist, info = get_yf_data(yf_symbol, start_date, end_date)
            if hist.empty:
                # try other plausible variants
                tried = []
                candidates = []
                # original input forms
                candidates.append(normalize_cse_ticker_for_yf(ticker_input))
                candidates.append(normalize_cse_ticker_for_cselk(ticker_input))  # just in case
                candidates.append(normalize_general_ticker(ticker_input))
                # try without .CM suffix
                t_no_cm = ticker_input.upper().replace('.CM', '').replace(' ', '.').replace('-', '.')
                candidates.append(t_no_cm)
                # unique
                candidates = [c for i, c in enumerate(candidates) if c and c not in candidates[:i]]
                for cand in candidates:
                    if cand in tried:
                        continue
                    tried.append(cand)
                    h, inf = get_yf_data(cand, start_date, end_date)
                    if not h.empty:
                        hist, info = h, inf
                        st.sidebar.success(f"Loaded history from Yahoo using symbol: {cand}")
                        break
                if hist.empty:
                    st.sidebar.error("Failed to load history from Yahoo for any tried symbol variants. See sidebar hints.")
    else:
        yf_symbol = normalize_general_ticker(ticker_input)
        hist, info = get_yf_data(yf_symbol, start_date, end_date)
        # try some common variants if yahoo failed
        if hist.empty:
            for cand in (ticker_input.upper(), ticker_input.replace('-', '.'), ticker_input.replace(' ', '.')):
                if cand == yf_symbol:
                    continue
                h, inf = get_yf_data(cand, start_date, end_date)
                if not h.empty:
                    hist, info = h, inf
                    st.sidebar.success(f"Loaded history from Yahoo using symbol: {cand}")
                    break

if hist.empty:
    st.error("⚠️ No data found for the provided ticker and date range. Please verify the ticker (try Normalize ticker), or upload a CSV/Excel. If using CSE data, consider installing cse.lk.")
    st.stop()

# Process indicators
hist = calculate_indicators(hist)
signal, reason = buy_sell_signal(hist)
score_strength = calculate_strength(hist)
insights = ai_market_analysis(hist, info, signal, reason)

# Tabs
tabs = ["📈 Chart", "📊 Indicators Hub", "🧠 AI Analysis & Strength Meter"]
if enable_news: tabs.append("📰 News")
tabs.append("🏢 Company Details")
t = st.tabs(tabs)

# Chart tab
with t[0]:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5,0.3,0.2])
    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close']), row=1,col=1)
    if 'SMA_20' in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index,y=hist['SMA_20'],name="SMA20"),row=1,col=1)
    if 'SMA_50' in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index,y=hist['SMA_50'],name="SMA50"),row=1,col=1)
    if 'MACD' in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index,y=hist['MACD'],name="MACD"),row=2,col=1)
    if 'MACD_signal' in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index,y=hist['MACD_signal'],name="Signal"),row=2,col=1)
    if 'RSI' in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index,y=hist['RSI'],name="RSI"),row=3,col=1)
    fig.update_layout(template="plotly_dark", height=800)
    # make plotly responsive
    st.plotly_chart(fig, use_container_width=True, config={'responsive': True})

# Indicators Hub tab
with t[1]:
    st.dataframe(hist.tail(15))
    if 'MACD' in hist.columns and 'MACD_signal' in hist.columns:
        st.line_chart(hist[['MACD','MACD_signal']])
    if 'RSI' in hist.columns:
        st.line_chart(hist['RSI'])

# AI Analysis tab
with t[2]:
    if signal == "BUY": st.success(f"✅ BUY — {reason}")
    elif signal == "SELL": st.error(f"❌ SELL — {reason}")
    else: st.warning(f"⚠️ HOLD — {reason}")
    for i in insights: st.write(f"- {i}")
    # Strength Meter Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_strength,
        title={'text': "Technical Strength"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 30], 'color': "red"},
                   {'range': [30, 70], 'color': "yellow"},
                   {'range': [70, 100], 'color': "green"}]}))
    st.plotly_chart(fig_gauge, use_container_width=True, config={'responsive': True})

# News tab
if enable_news and "📰 News" in tabs:
    news_tab_index = tabs.index("📰 News")
    with t[news_tab_index]:
        if news_source == "Yahoo Finance":
            # determine best yahoo symbol we used (prefer yf_symbol if defined)
            yahoo_try_symbol = None
            if 'yf_symbol' in locals():
                yahoo_try_symbol = locals().get('yf_symbol')
            else:
                yahoo_try_symbol = normalize_general_ticker(ticker_input)
            try:
                for n in fetch_yahoo_news(yahoo_try_symbol):
                    st.write(f"- [{n.get('title')}]({n.get('link')})")
            except Exception:
                st.info("Unable to fetch Yahoo news for this symbol.")
        elif news_source == "EconomyNext business":
            for n in scrape_site("https://economynext.com/category/business/"):
                st.write(f"- {n[0]} ({n[1]})")
        elif news_source == "EconomyNext market":
            for n in scrape_site("https://economynext.com/markets/"):
                st.write(f"- {n[0]} ({n[1]})")

# Company Details tab
with t[-1]:
    if platform.startswith("CSE"):
        cse_symbol = normalize_cse_ticker_for_cselk(ticker_input)
        st.write(f"Fetching CSE details for {cse_symbol} using cse.lk package (if installed)...")
        details = get_cse_company_details(cse_symbol)
        if isinstance(details, dict):
            if not details:
                st.warning("No details returned from cse.lk.")
            elif 'error' in details:
                st.error(details['error'])
                # give user a hint on how to install
                st.info("If you want to use cse.lk, install it in your environment: pip install cse.lk")
            else:
                # display dictionary-like details
                for k, v in details.items():
                    st.write(f"**{k}:** {v}")
        else:
            # if the library returned an object, try to pretty-print attributes
            try:
                attrs = {a: getattr(details, a) for a in dir(details) if not a.startswith("_")}
                for k, v in attrs.items():
                    st.write(f"**{k}:** {v}")
            except Exception:
                st.write(details)
    else:
        st.write(f"Showing Yahoo Finance summary for {normalize_general_ticker(ticker_input)}...")
        if info:
            # show a few key fields from yfinance info
            for k in ["longName", "sector", "industry", "marketCap", "previousClose", "open", "fiftyTwoWeekHigh", "fiftyTwoWeekLow"]:
                if k in info:
                    st.write(f"**{k}:** {info.get(k)}")
        else:
            st.warning("No company info found via Yahoo Finance.")
