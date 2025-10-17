import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta, date
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="AI + CSE Stock Dashboard", layout="wide")

# ====================== HELPERS =======================
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

# ============= Technical Indicators Functions =============
def calculate_indicators(df):
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta>0, 0).rolling(14).mean()
    loss = -delta.where(delta<0, 0).rolling(14).mean()
    rs = gain/loss
    df['RSI'] = 100 - (100/(1+rs))
    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_upper'] = bb_mid + (bb_std*2)
    df['BB_lower'] = bb_mid - (bb_std*2)
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

# ================ ML MODEL =================
class StockML:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def prepare_data(self, df):
        df['Returns'] = df['Close'].pct_change()
        df['Close_lag1'] = df['Close'].shift(1)
        df['Close_lag2'] = df['Close'].shift(2)
        df['RSI'] = df['RSI'].fillna(method='bfill')
        df = df.dropna()
        X = df[['Close_lag1', 'Close_lag2', 'RSI', 'MACD', 'MACD_signal']]
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

# =================== UI =====================
st.sidebar.title("📊 Stock Dashboard Controls")
ticker_input = st.sidebar.text_input("Stock Ticker", "AAPL")
use_cse = st.sidebar.checkbox("🇱🇰 Use CSE format (.CM)", value=False)
start_date = st.sidebar.date_input("Start Date", date(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())
analysis_mode = st.sidebar.selectbox("Analysis Mode", ["AI Technical Dashboard", "Buy/Sell Signal Chart"])
show_prediction = st.sidebar.checkbox("Show ML Prediction", value=True)

# Adjust ticker for CSE if needed
symbol = normalize_cse_ticker(ticker_input) if use_cse else ticker_input.upper()

hist, info = get_yf_data(symbol, start_date, end_date)

if hist.empty:
    st.error("No data found for given ticker/range")
    st.stop()

# Calculate indicators
hist = calculate_indicators(hist)
hist, pivots = zigzag_with_signals(hist)
signal, reason = buy_sell_signal(hist)

# ========== Analysis Modes ==========
if analysis_mode == "AI Technical Dashboard":
    st.subheader(f"📈 {symbol} Technical Analysis")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.5,0.3,0.2],
                        subplot_titles=(f"{symbol} Price", "MACD", "RSI"))
    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                 low=hist['Low'], close=hist['Close']), row=1,col=1)
    fig.add_trace(go.Scatter(x=hist.index,y=hist['SMA_20'],name="SMA20"),row=1,col=1)
    fig.add_trace(go.Scatter(x=hist.index,y=hist['SMA_50'],name="SMA50"),row=1,col=1)
    fig.add_trace(go.Scatter(x=hist.index,y=hist['MACD'],name="MACD"),row=2,col=1)
    fig.add_trace(go.Scatter(x=hist.index,y=hist['MACD_signal'],name="Signal"),row=2,col=1)
    fig.add_trace(go.Scatter(x=hist.index,y=hist['RSI'],name="RSI"),row=3,col=1)
    fig.update_layout(template="plotly_dark",height=800)
    st.plotly_chart(fig,use_container_width=True)

else:
    st.subheader(f"📊 Buy/Sell Signals for {symbol}")
    fig = go.Figure(go.Candlestick(x=hist.index,open=hist["Open"],high=hist["High"],low=hist["Low"],close=hist["Close"]))
    for d,p,act in pivots:
        fig.add_trace(go.Scatter(x=[d],y=[p],mode="markers+text",text=[act.upper()],
                                 marker=dict(color="green" if act=="buy" else "red", size=10)))
    st.plotly_chart(fig,use_container_width=True)
    st.markdown(f"**Recommendation:** {signal} — {reason}")

# ========== ML Prediction ==========
if show_prediction:
    ml = StockML()
    res = ml.train(hist)
    if res:
        score, last_feat = res
        pred = ml.predict_next(last_feat)
        st.success(f"ML predicts next close: ${pred:.2f} — Model R²: {score:.2%}")
    else:
        st.warning("Insufficient data for ML prediction")

# ====== EXTRA INFO =======
st.markdown("---")
st.write("### Company Info")
if info:
    st.write(info.get("longName",symbol))
    st.write(f"Sector: {info.get('sector','N/A')}")
    st.write(f"Market Cap: {info.get('marketCap','N/A')}")
else:
    st.write("No company info available.")
