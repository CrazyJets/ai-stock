import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import date
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re
import warnings
warnings.filterwarnings('ignore')

# DuckDuckGo safe import
try:
    from duckduckgo_search import ddg_news
    duckduckgo_available = True
except ImportError:
    duckduckgo_available = False

st.set_page_config(page_title="Enhanced AI + CSE Stock Dashboard", layout="wide")

# ---------- AUTO-DETECT CSE ----------
def detect_and_normalize_ticker(ticker):
    if re.match(r"^[A-Z]{3,4}\.[A-Z]\d{4}$", ticker.upper()):
        return ticker.upper() + ".CM"
    return ticker.upper()

# ---------- Fetch data ----------
@st.cache_data
def get_yf_data(symbol, start, end):
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(start=start, end=end)
        hist = pd.DataFrame(hist).copy()
        info = dict(tk.info or {})
        return hist, info
    except Exception:
        return pd.DataFrame(), {}

# ---------- Indicators ----------
def calculate_indicators(df):
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    delta = df['Close'].diff()
    gain = delta.where(delta>0, 0).rolling(14).mean()
    loss = -delta.where(delta<0, 0).rolling(14).mean()
    rs = gain/loss
    df['RSI'] = 100 - (100/(1+rs))
    return df

def buy_sell_signal(df):
    latest_macd = df['MACD'].iloc[-1]
    latest_signal = df['MACD_signal'].iloc[-1]
    latest_rsi = df['RSI'].iloc[-1]
    if latest_macd > latest_signal and latest_rsi < 70:
        return "BUY", "MACD crossover + RSI under 70"
    elif latest_macd < latest_signal and latest_rsi > 30:
        return "SELL", "MACD down + RSI above 30"
    return "HOLD", "No strong signal"

# ---------- AI ML ----------
class StockML:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    def prepare(self, df):
        df = df.dropna()
        X = df[['Close','SMA_20','SMA_50','RSI','MACD','MACD_signal']]
        y = df['Close'].shift(-1).dropna()
        return X.iloc[:-1], y.iloc[:-1]
    def train(self, df):
        X, y = self.prepare(df)
        if len(X) < 30: return None
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
        self.model.fit(self.scaler.fit_transform(X_train), y_train)
        score = self.model.score(self.scaler.transform(X_test), y_test)
        return score, X.iloc[-1:]
    def predict_next(self, features):
        return self.model.predict(self.scaler.transform(features))[0]

# ---------- News Scraping ----------
@st.cache_data
def scrape_economynext_business():
    try:
        res = requests.get("https://economynext.com/category/business/", timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        return [(a.text.strip(), a['href']) for a in soup.find_all("a", href=True) if a.text.strip()]
    except:
        return []

@st.cache_data
def scrape_economynext_markets():
    try:
        res = requests.get("https://economynext.com/markets/", timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        return [(a.text.strip(), a['href']) for a in soup.find_all("a", href=True) if a.text.strip()]
    except:
        return []

@st.cache_data
def scrape_bartleet_research():
    try:
        res = requests.get("https://research.bartleetreligare.com/", timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        return [(a.text.strip(), a['href']) for a in soup.select("a") if a.get('href')]
    except:
        return []

def fetch_yahoo_news(symbol):
    try:
        t = yf.Ticker(symbol)
        return [(n['title'], n['link']) for n in t.news]
    except:
        return []

def fetch_ddg_news(query):
    if not duckduckgo_available: return []
    try:
        return [(n.get('title'), n.get('url')) for n in ddg_news(query, max_results=7)]
    except:
        return []

# ---------- Sidebar ----------
st.sidebar.title("Controls 🛠")
ticker = st.sidebar.text_input("Ticker", "SAMP.N0000")
ticker = detect_and_normalize_ticker(ticker)
start_date = st.sidebar.date_input("Start Date", date(2024,1,1))
end_date = st.sidebar.date_input("End Date", date.today())
show_pred = st.sidebar.checkbox("Show ML Prediction", True)
enable_news = st.sidebar.checkbox("News Tab", True)
news_source = st.sidebar.radio("Select News Source", ["DuckDuckGo","Yahoo Finance","EconomyNext All"])

# ---------- Data Load ----------
hist, info = get_yf_data(ticker, start_date, end_date)
if hist.empty:
    st.error("No data found")
    st.stop()

hist = calculate_indicators(hist)
signal, reason = buy_sell_signal(hist)

# ---------- Tabs ----------
tabs = ["Chart", "Indicators & Charts", "AI Analysis"]
if enable_news: tabs.append("News")
t_objs = st.tabs(tabs)

# Chart Tab
with t_objs[0]:
    st.subheader(f"📈 Price Chart for {ticker}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close']))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name="SMA20"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name="SMA50"))
    fig.update_layout(template="plotly_dark", height=700)
    st.plotly_chart(fig, use_container_width=True)

# Indicators Tab
with t_objs[1]:
    st.subheader("MACD Chart 📊")
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name="MACD"))
    macd_fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD_signal'], name="Signal"))
    macd_fig.add_trace(go.Bar(x=hist.index, y=hist['MACD_hist'], name="Hist", marker_color="green"))
    st.plotly_chart(macd_fig, use_container_width=True)

    st.subheader("RSI Chart 📈")
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name="RSI", line_color="orange"))
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
    st.plotly_chart(rsi_fig, use_container_width=True)

    st.dataframe(hist.tail(15))

# AI Analysis Tab
with t_objs[2]:
    st.markdown("### AI Recommendation & Prediction")
    if signal=="BUY":
        st.success(f"📈 BUY — {reason}")
    elif signal=="SELL":
        st.error(f"📉 SELL — {reason}")
    else:
        st.warning(f"➖ HOLD — {reason}")

    if show_pred:
        ml = StockML()
        res = ml.train(hist)
        if res:
            score, last_feat = res
            pred_price = ml.predict_next(last_feat)
            icon = "📈" if pred_price>hist['Close'].iloc[-1] else "📉"
            conf = "High" if score>0.8 else "Medium" if score>0.6 else "Low"
            st.info(f"{icon} Predicted Next Close: {pred_price:.2f} | Confidence: {conf} ({score:.2%})")

# News Tab
if enable_news:
    with t_objs[3]:
        st.subheader("Market News 📰")
        if news_source=="DuckDuckGo":
            data = fetch_ddg_news(f"{ticker} stock")
        elif news_source=="Yahoo Finance":
            data = fetch_yahoo_news(ticker)
        else:
            data = scrape_economynext_business() + scrape_economynext_markets() + scrape_bartleet_research()
        st.write("**Common News**")
        for title,link in data[:10]:
            st.write(f"- [{title}]({link})")
        st.write("**Selected Share Related News**")
        sel_related = [n for n in data if ticker.split('.')[0].lower() in title.lower()]
        for title,link in sel_related:
            st.write(f"- [{title}]({link})")
