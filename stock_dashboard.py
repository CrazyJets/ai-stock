import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from datetime import date
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

# PDF extraction
from PyPDF2 import PdfReader

# cloudscraper is helpful to handle common Cloudflare checks in a non-invasive manner.
# If you don't have it in your environment, install: pip install cloudscraper
import cloudscraper

warnings.filterwarnings('ignore')

st.set_page_config(page_title="💹 AI + CSE Stock Dashboard", layout="wide")

# ---------------- Helper Functions -----------------
def normalize_cse_ticker(ticker: str) -> str:
    if ticker and not ticker.upper().endswith(".CM"):
        return ticker.upper() + ".CM"
    return ticker.upper()

def convert_for_stockanalysis(cse_ticker: str) -> str:
    """Convert Yahoo Finance CSE ticker to StockAnalysis format"""
    return cse_ticker.replace(".CM", "").replace("-", ".")

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
    df['Vol_SMA'] = df['Volume'].rolling(20).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(10).std()*np.sqrt(252)*100
    return df

def buy_sell_signal(df):
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
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    pc = (latest['Close'] - prev['Close']) / prev['Close'] * 100
    vol_ratio = latest['Volume'] / latest['Vol_SMA'] if latest['Vol_SMA'] else 1
    insights = [f"📊 Change: {pc:+.2f}% → {signal} — {reason}"]
    if latest['RSI'] > 70: insights.append(f"⚠️ RSI {latest['RSI']:.1f} Overbought risk")
    elif latest['RSI'] < 30: insights.append(f"💡 RSI {latest['RSI']:.1f} Oversold opportunity")
    if latest['MACD'] > latest['MACD_signal']: insights.append("📈 MACD bullish momentum")
    else: insights.append("📉 MACD bearish momentum")
    if vol_ratio > 1.5: insights.append("🔥 High volume confirms strong move")
    elif vol_ratio < 0.7: insights.append("📊 Weak volume")
    return insights

# Technical Strength Meter
def calculate_strength(df):
    latest = df.iloc[-1]
    score = 50
    if latest['RSI'] > 70: score -= 20
    elif latest['RSI'] < 30: score += 20
    if latest['MACD'] > latest['MACD_signal']: score += 15
    else: score -= 15
    # Bollinger position
    if latest['Close'] > latest['BB_upper']: score -= 10
    elif latest['Close'] < latest['BB_lower']: score += 10
    # Volume
    vol_ratio = latest['Volume'] / latest['Vol_SMA'] if latest['Vol_SMA'] else 1
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
SCRAPER = cloudscraper.create_scraper()  # handles many common Cloudflare checks

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
    lower = text.lower()
    return any(s in lower for s in block_signals)

def get_html(url, use_scraper=True, timeout=15):
    """Return (text, final_url, status_code) or (None, url, status) if blocked/fails."""
    headers = {"User-Agent": "ai-stock-dashboard/1.0 (+https://github.com/CrazyJets)"}
    try:
        if use_scraper:
            r = SCRAPER.get(url, timeout=timeout, headers=headers)
        else:
            r = requests.get(url, timeout=timeout, headers=headers)
        text = r.text or ""
        if is_blocked_by_cf(text, r.status_code):
            return None, r.url, r.status_code
        return text, r.url, r.status_code
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

# News & StockAnalysis scrapers
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

def scrape_stockanalysis_sections(symbol_for_stockanalysis):
    """Fetch multiple StockAnalysis pages for a given symbol and extract elements with data-title='overall' or fallbacks.

    Uses the scraper that can handle Cloudflare checks. If the site appears blocked we fall back to yfinance info.
    """
    base = f"https://stockanalysis.com/quote/cose/{symbol_for_stockanalysis}/"
    paths = {'overall': base,
             'financials': base + 'financials/',
             'dividend': base + 'dividend/',
             'company': base + 'company/',
             'statistics': base + 'statistics/'}
    result = {}
    for name, url in paths.items():
        text, final_url, status = get_html(url, use_scraper=True)
        if text is None:
            # blocked or not accessible
            result[name] = {"error": f"Blocked or unavailable (status={status}). Falling back to Yahoo Finance where possible."}
            continue
        try:
            soup = BeautifulSoup(text, 'html.parser')
            nodes = soup.find_all(attrs={"data-title": "overall"})
            if nodes:
                texts = []
                for n in nodes:
                    texts.append(' '.join(n.stripped_strings))
                result[name] = '\n\n'.join(texts)
                continue
            # Secondary: snapshot items
            summary = {}
            h1 = soup.find('h1')
            if h1:
                summary['Name'] = h1.get_text(strip=True)
            smalls = soup.find_all('small')
            if smalls:
                summary['Description'] = smalls[0].get_text(strip=True)
            divs = soup.find_all('div', class_=re.compile(r'snapshot__item|snapshot-item'))
            for d in divs:
                label = d.find('div', class_=re.compile(r'label|snapshot__label'))
                value = d.find('div', class_=re.compile(r'value|snapshot__value'))
                if label and value:
                    summary[label.get_text(strip=True)] = value.get_text(strip=True)
            if summary:
                result[name] = summary
                continue
            # Final fallback: first paragraph or empty string
            p = soup.find('p')
            result[name] = p.get_text(strip=True) if p else ''
        except Exception as e:
            result[name] = {"error": f"Parsing error: {e}"}
    # If everything blocked, try to provide basic yfinance info for 'overall'
    if all(isinstance(v, dict) and ('error' in v) for v in result.values()):
        try:
            ticker = yf.Ticker(symbol_for_stockanalysis)
            info = dict(ticker.info or {})
            result['yf_fallback'] = info
        except Exception:
            result['yf_fallback'] = {}
    return result

# ---------------- Bartleet Religare Reports scraping ----------------
BARTLEET_BASE = "https://research.bartleetreligare.com/reports?category=market-updates"

@st.cache_data(show_spinner=False)
def download_bytes(url, use_scraper=True, timeout=20):
    headers = {"User-Agent": "ai-stock-dashboard/1.0 (+https://github.com/CrazyJets)"}
    try:
        if use_scraper:
            r = SCRAPER.get(url, timeout=timeout, headers=headers)
        else:
            r = requests.get(url, timeout=timeout, headers=headers)
        if r.status_code != 200:
            return None, r.status_code
        return r.content, r.status_code
    except Exception:
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            if r.status_code != 200:
                return None, r.status_code
            return r.content, r.status_code
        except Exception:
            return None, None

@st.cache_data(show_spinner=False)
def extract_pdf_text_from_bytes(pdf_bytes):
    if not pdf_bytes:
        return ""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        text_parts = []
        for pg in reader.pages:
            try:
                t = pg.extract_text()
                if t:
                    text_parts.append(t)
            except Exception:
                continue
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"[PDF extraction error: {e}]"

@st.cache_data(show_spinner=False)
def fetch_bartleet_reports_list():
    # Fetch the reports listing page and extract links to PDFs or report detail pages
    text, final_url, status = get_html(BARTLEET_BASE)
    results = []
    if not text:
        return results
    soup = BeautifulSoup(text, "html.parser")
    # Look for anchor tags that directly point to PDFs
    for a in soup.find_all("a", href=True):
        href = urljoin(final_url, a['href'])
        title = a.get_text(strip=True) or href
        if re.search(r"\.pdf(\?|$)", href.lower()):
            results.append({"title": title, "url": href, "type": "pdf"})
        else:
            # Some report listings link to a report page. We'll store them too to be checked later.
            # Heuristic: if URL path contains '/reports/' or '/report/', consider it a report page.
            if "/report" in href or "/reports" in href:
                results.append({"title": title, "url": href, "type": "page"})
    # De-duplicate by URL while preserving order
    seen = set()
    deduped = []
    for r in results:
        if r["url"] not in seen:
            deduped.append(r)
            seen.add(r["url"])
    return deduped

@st.cache_data(show_spinner=False)
def resolve_report_to_pdf(report_entry):
    """Given an entry (pdf or page), return pdf_url (or None) and extracted text (or empty)."""
    if report_entry["type"] == "pdf":
        pdf_url = report_entry["url"]
        content, status = download_bytes(pdf_url)
        if content:
            text = extract_pdf_text_from_bytes(content)
            return pdf_url, text
        return pdf_url, ""
    # If it's a page, fetch it and look for PDF links or inline viewers
    text, final_url, status = get_html(report_entry["url"])
    if not text:
        return None, ""
    soup = BeautifulSoup(text, "html.parser")
    # Look for direct PDF link in page
    for a in soup.find_all("a", href=True):
        href = urljoin(final_url, a['href'])
        if re.search(r"\.pdf(\?|$)", href.lower()):
            content, stcode = download_bytes(href)
            if content:
                return href, extract_pdf_text_from_bytes(content)
    # Look for iframe that might embed a PDF
    iframe = soup.find("iframe", src=True)
    if iframe:
        src = urljoin(final_url, iframe['src'])
        if re.search(r"\.pdf(\?|$)", src.lower()):
            content, stcode = download_bytes(src)
            if content:
                return src, extract_pdf_text_from_bytes(content)
    # No PDF found
    return None, ""

# ---------------- Sidebar controls ----------------
st.sidebar.title("📊 Dashboard Controls")
upload_data = st.sidebar.file_uploader("📤 Upload CSV/Excel", type=["csv", "xlsx"])
ticker_input = st.sidebar.text_input("Ticker", "WIND-N0000.CM")  # Yahoo format
use_cse = st.sidebar.checkbox("🇱🇰 Use CSE format (.CM)", value=False)
start_date = st.sidebar.date_input("Start Date", date(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())
show_prediction = st.sidebar.checkbox("🔮 Show ML Prediction", value=True)
enable_news = st.sidebar.checkbox("📰 Show Market News", value=True)
news_source = st.sidebar.selectbox("Choose News Source", ["Yahoo Finance", "EconomyNext business", "EconomyNext market", "Bartleet Religare"])

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

# Load data
if upload_data is not None:
    if upload_data.name.endswith(".csv"):
        hist = pd.read_csv(upload_data, parse_dates=[0])
    else:
        hist = pd.read_excel(upload_data, parse_dates=[0])
    hist = hist.set_index(hist.columns[0])
    info = {}
else:
    symbol = normalize_cse_ticker(ticker_input) if use_cse else ticker_input.upper()
    hist, info = get_yf_data(symbol, start_date, end_date)

if hist.empty:
    st.error("⚠️ No data found")
    st.stop()

# Process indicators
hist = calculate_indicators(hist)
signal, reason = buy_sell_signal(hist)
score_strength = calculate_strength(hist)
insights = ai_market_analysis(hist, info, signal, reason)

# Tabs
tabs = ["📈 Chart", "📊 Indicators Hub", "🧠 AI Analysis & Strength Meter"]
if enable_news: tabs.append("📰 News & Bartleet Religare")
tabs.append("🏢 Company Details")
t = st.tabs(tabs)

# Chart tab
with t[0]:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5,0.3,0.2])
    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close']), row=1,col=1)
    fig.add_trace(go.Scatter(x=hist.index,y=hist['SMA_20'],name="SMA20"),row=1,col=1)
    fig.add_trace(go.Scatter(x=hist.index,y=hist['SMA_50'],name="SMA50"),row=1,col=1)
    fig.add_trace(go.Scatter(x=hist.index,y=hist['MACD'],name="MACD"),row=2,col=1)
    fig.add_trace(go.Scatter(x=hist.index,y=hist['MACD_signal'],name="Signal"),row=2,col=1)
    fig.add_trace(go.Scatter(x=hist.index,y=hist['RSI'],name="RSI"),row=3,col=1)
    fig.update_layout(template="plotly_dark", height=800)
    # make plotly responsive
    st.plotly_chart(fig, use_container_width=True, config={'responsive': True})

# Indicators Hub tab
with t[1]:
    st.dataframe(hist.tail(15))
    st.line_chart(hist[['MACD','MACD_signal']])
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

# News tab (including Bartleet Religare)
if enable_news:
    with t[3]:
        if news_source == "Yahoo Finance":
            for n in fetch_yahoo_news(ticker_input):
                st.write(f"- [{n.get('title')}]({n.get('link')})")
        elif news_source == "EconomyNext business":
            for n in scrape_site("https://economynext.com/category/business/"):
                st.write(f"- {n[0]} ({n[1]})")
        elif news_source == "EconomyNext market":
            for n in scrape_site("https://economynext.com/markets/"):
                st.write(f"- {n[0]} ({n[1]})")
        elif news_source == "Bartleet Religare":
            st.write("Discovering Bartleet Religare market-updates reports (public PDFs where available)...")
            reports = fetch_bartleet_reports_list()
            if not reports:
                st.warning("No reports discovered or page blocked/unavailable.")
            else:
                for r in reports:
                    title = r.get("title") or r.get("url")
                    st.markdown(f"**{title}**")
                    st.write(r.get("url"))
                    pdf_url, extracted_text = resolve_report_to_pdf(r)
                    if pdf_url:
                        # provide a simple download link + show first part of the text in an expander
                        st.markdown(f"- PDF: {pdf_url}")
                        with st.expander("Show extracted text (first 8000 chars)"):
                            st.text(extracted_text[:8000] if extracted_text else "No text extracted.")
                    else:
                        st.info("No downloadable PDF found for this report (might be behind a viewer).")

# Company Details tab
with t[-1]:
    stockanalysis_symbol = convert_for_stockanalysis(ticker_input)
    st.write(f"Fetching data from StockAnalysis for {stockanalysis_symbol}...")
    company_info_sections = scrape_stockanalysis_sections(stockanalysis_symbol)
    if company_info_sections:
        for section, content in company_info_sections.items():
            st.header(section.capitalize())
            if isinstance(content, dict):
                if content:
                    for k,v in content.items():
                        st.write(f"**{k}:** {v}")
                else:
                    st.write("No data found.")
            else:
                st.write(content or "No data found.")
    else:
        st.warning("No company info found.")
