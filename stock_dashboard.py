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
warnings.filterwarnings('ignore')

# Safe import of DuckDuckGo
try:
    from duckduckgo_search import ddg_news
    duckduckgo_available = True
except ImportError:
    duckduckgo_available = False

st.set_page_config(page_title="💹 AI + CSE Stock Dashboard", layout="wide")

# ---------------- Helper Functions -----------------
def normalize_cse_ticker(ticker: str) -> str:
    if ticker and not ticker.upper().endswith(".CM"):
        return ticker.upper() + ".CM"
    return ticker.upper()

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

# Calculate Indicators
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

def zigzag_with_signals(df, threshold=0.03):
    df = df.copy()
    df['ZigZag'] = np.nan
    pivots = []
    last_pivot = df['Close'].iloc[0]
    direction = None
    for i in range(1, len(df)):
        price = df['Close'].iat[i]
        change = (price-last_pivot)/last_pivot
        if (direction != "up") and (change > threshold):
            direction = "up"
            last_pivot = price
            df.at[df.index[i], "ZigZag"] = price
            pivots.append((df.index[i], price, "buy"))
        elif (direction != "down") and (change < -threshold):
            direction = "down"
            last_pivot = price
            df.at[df.index[i], "ZigZag"] = price
            pivots.append((df.index[i], price, "sell"))
    return df, pivots

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

# AI Market Insights
def ai_market_analysis(df, info, signal, reason):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    pc = (latest['Close'] - prev['Close']) / prev['Close'] * 100
    rsi = latest['RSI']
    macd = latest['MACD']
    macd_signal = latest['MACD_signal']
    bb_upper = latest['BB_upper']
    bb_lower = latest['BB_lower']
    vol_ratio = latest['Volume'] / latest['Vol_SMA'] if latest['Vol_SMA'] else 1

    insights = [f"📊 Today change: {pc:+.2f}% → {signal} — {reason}"]
    if rsi > 70: insights.append(f"⚠️ RSI {rsi:.1f} → Overbought risk")
    elif rsi < 30: insights.append(f"💡 RSI {rsi:.1f} → Oversold opportunity")
    if latest['Close'] > bb_upper: insights.append("🚀 Price above Bollinger Upper → Possible reversal")
    elif latest['Close'] < bb_lower: insights.append("📉 Price below Bollinger Lower → Potential bounce")
    if macd > macd_signal: insights.append("📈 MACD bullish momentum")
    else: insights.append("📉 MACD bearish momentum")
    if vol_ratio > 1.5: insights.append("🔥 High volume confirms strong move")
    elif vol_ratio < 0.7: insights.append("📊 Weak volume, possible false move")
    mcap = info.get('marketCap', 0)
    if mcap > 1e11: insights.append("🏢 Large-cap stability")
    elif mcap > 1e9: insights.append("🏭 Mid-cap balance")
    else: insights.append("📈 Small-cap volatility")
    return insights

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

# News scraping
def scrape_site(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        return [a.text.strip() for a in soup.find_all("a", href=True) if a.text.strip()]
    except Exception:
        return []

def fetch_duckduckgo_news(query):
    if not duckduckgo_available: return []
    try:
        return ddg_news(query, max_results=7)
    except Exception:
        return []

def fetch_yahoo_news(symbol):
    try:
        t = yf.Ticker(symbol)
        return t.news
    except Exception:
        return []

# Sidebar controls
st.sidebar.title("📊 Controls")
upload_data = st.sidebar.file_uploader("📤 Upload CSV/Excel", type=["csv", "xlsx"])
ticker_input = st.sidebar.text_input("Ticker", "AAPL")
use_cse = st.sidebar.checkbox("🇱🇰 Use CSE format (.CM)", value=False)
start_date = st.sidebar.date_input("Start Date", date(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())
show_prediction = st.sidebar.checkbox("🔮 Show ML Prediction", value=True)
enable_news = st.sidebar.checkbox("📰 Show Market News", value=True)
news_source = st.sidebar.selectbox("Choose News Source", ["DuckDuckGo", "Yahoo Finance", "EconomyNext business", "EconomyNext market", "Bartleet Religare"])

# Data load
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

# Process indicators & signals
hist = calculate_indicators(hist)
hist, pivots = zigzag_with_signals(hist)
signal, reason = buy_sell_signal(hist)
insights = ai_market_analysis(hist, info, signal, reason)

# Tabs
tabs = ["📈 Chart", "📊 Indicators Hub", "🧠 AI Analysis & Signals"]
if enable_news: tabs.append("📰 News")
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
    st.plotly_chart(fig, use_container_width=True)

# Indicators Hub tab
with t[1]:
    st.subheader("📊 Latest Indicators Data")
    st.dataframe(hist.tail(15))
    st.subheader("📈 Bollinger Bands")
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=hist.index, y=hist['BB_upper'], line=dict(color='grey'), name='BB Upper'))
    fig_bb.add_trace(go.Scatter(x=hist.index, y=hist['BB_lower'], line=dict(color='grey'), name='BB Lower', fill='tonexty', fillcolor='rgba(200,200,200,0.1)'))
    fig_bb.add_trace(go.Scatter(x=hist.index, y=hist['Close'], line=dict(color='blue'), name='Close'))
    st.plotly_chart(fig_bb, use_container_width=True)
    st.subheader("🌡 Volatility Heat Map (%)")
    fig_vol = go.Figure(go.Bar(x=hist.index, y=hist['Volatility'], marker_color=hist['Volatility'], marker=dict(colorscale='Reds')))
    st.plotly_chart(fig_vol, use_container_width=True)
    st.subheader("📊 MACD Histogram")
    fig_macd = go.Figure(go.Bar(x=hist.index, y=hist['MACD_histogram'], marker_color=['green' if v>=0 else 'red' for v in hist['MACD_histogram']]))
    st.plotly_chart(fig_macd, use_container_width=True)
    st.subheader("📈 RSI Color Zones")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], line=dict(color='purple')))
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.2)
    fig_rsi.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.2)
    st.plotly_chart(fig_rsi, use_container_width=True)
    st.subheader("📦 Volume Trend")
    st.line_chart(hist[['Volume','Vol_SMA']])

# AI Analysis tab
with t[2]:
    if signal == "BUY":
        st.success(f"✅ BUY — {reason}")
    elif signal == "SELL":
        st.error(f"❌ SELL — {reason}")
    else:
        st.warning(f"⚠️ HOLD — {reason}")
    for insight in insights:
        st.write(f"- {insight}")
    if show_prediction:
        ml = StockML()
        res = ml.train(hist)
        if res:
            score, last_feat = res
            pred = ml.predict_next(last_feat)
            conf = "High" if score > 0.8 else "Medium" if score > 0.6 else "Low"
            st.info(f"🤖 Predicted close: {pred:.2f} | Confidence: {conf} ({score:.2%})")

# News tab
if enable_news:
    with t[3]:
        st.subheader("📰 Common Market News")
        if news_source == "DuckDuckGo":
            for n in fetch_duckduckgo_news(f"{ticker_input} stock news"):
                st.write(f"- [{n.get('title')}]({n.get('url')})")
        elif news_source == "Yahoo Finance":
            for n in fetch_yahoo_news(ticker_input):
                st.write(f"- [{n.get('title')}]({n.get('link')})")
        elif news_source == "EconomyNext business":
            for n in scrape_site("https://economynext.com/category/business/"):
                st.write(f"- {n}")
        elif news_source == "EconomyNext market":
            for n in scrape_site("https://economynext.com/markets/"):
                st.write(f"- {n}")
        elif news_source == "Bartleet Religare":
            for n in scrape_site("https://research.bartleetreligare.com/"):
                st.write(f"- {n}")
        st.subheader("🎯 Selected Share Related News")
        ticker_upper = ticker_input.upper()
        headlines = scrape_site("https://economynext.com/category/business/") + scrape_site("https://economynext.com/markets/")
        filtered = [h for h in headlines if ticker_upper in h.upper()]
        if filtered:
            for h in filtered:
                st.write(f"- {h}")
        else:
            st.write("No related news found.")
