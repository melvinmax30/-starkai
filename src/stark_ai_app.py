# === stark_ai_app.py (StarkAI v10.0 Quantum Upgrade - Final Fix) ===

import streamlit as st
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from textblob import TextBlob
import pyttsx3
import speech_recognition as sr
import openai
import plotly.graph_objects as go
import smtplib
import pandas as pd
from email.mime.text import MIMEText


# --------------- GPT Setup ---------------
openai.api_key = st.secrets["openai_api_key"]

# --------------- Jarvis Voice Output ---------------
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        engine.say(text)
        engine.runAndWait()
    except Exception:
        st.warning("🔇 Voice error: eSpeak is likely not supported on cloud deployment.")

# --------------- Voice Input ---------------
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("🎙️ Listening... Speak a ticker symbol.")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            query = recognizer.recognize_google(audio)
            return query.upper()
        except sr.UnknownValueError:
            st.warning("❗ Couldn’t understand audio.")
        except sr.RequestError as e:
            st.error(f"Speech recognition error: {e}")

# --------------- Sentiment Analyzer ---------------
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return "🟢 Positive" if polarity > 0.2 else "🔴 Negative" if polarity < -0.2 else "🟡 Neutral"

# --------------- News Scraper ---------------
def scrape_news(ticker):
    url = f"https://news.google.com/search?q={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("a", class_="DY5T1d", limit=5)
    return [article.text for article in articles]

# --------------- Stock Predictor ---------------
def predict_stock(ticker):
    data = yf.download(ticker, period="5d", interval="1h", auto_adjust=True)

    # Catch empty or invalid data
    if data.empty or "Close" not in data or data['Close'].isna().all():
        return None

    # Safely extract the latest values
    latest_close = data['Close'].iloc[-1]
    avg_close = data['Close'].mean()
    volume = data['Volume'].iloc[-1]

    # Double-check for NaNs or wrong types
    if pd.isna(latest_close) or pd.isna(avg_close):
        return None

    # FIXED LINE
    trend = "Up" if float(latest_close) > float(avg_close) else "Down"

    return {
        "latest_close": round(float(latest_close), 2),
        "average_close": round(float(avg_close), 2),
        "volume": int(volume),
        "trend": trend
    }



# --------------- Recommendation Engine ---------------
def give_recommendation(pred, ticker):
    if not pred:
        return "❌ No data available."
    if pred["trend"] == "Up" and pred["volume"] > 1_000_000:
        return f"🚀 Strong Buy for {ticker} – rising price & high volume."
    elif pred["trend"] == "Up":
        return f"📈 Mild Buy for {ticker} – upward trend but volume low."
    else:
        return f"⚠️ Caution: {ticker} is trending down. Hold or reassess."

# --------------- GPT Market Commentary ---------------
def gpt_commentary(ticker, headlines, prediction):
    if not prediction:
        return "⚠️ GPT cannot generate commentary without valid market data."
    prompt = f"""
You are an AI financial analyst. Analyze the current market sentiment for {ticker}:

Trend: {prediction['trend']}
Volume: {prediction['volume']}
Recent News: {" | ".join(headlines)}

Return a smart and short recommendation in plain English.
"""
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ GPT error: {e}"

# --------------- Simulated Trading ---------------
def simulate_trade(ticker, recommendation, enabled):
    if not enabled:
        return "🛑 Trading disabled (Kill Switch ON)."
    if "Buy" in recommendation:
        return f"✅ Simulated BUY for {ticker}"
    elif "Caution" in recommendation:
        return f"⏸️ HOLD {ticker}"
    return f"🚫 No trade executed for {ticker}"

# --------------- Telegram Alert ---------------
def send_telegram(msg):
    try:
        token = st.secrets["7685701197:AAFQ7zncLQINzzWChGU3b96AB378hTbLMKM"]
        chat_id = st.secrets["1978718953"]
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": msg})
    except:
        pass

# --------------- Email Alert ---------------
def send_email(subject, body):
    try:
        sender = st.secrets["melvin_max10@hotmail.com"]
        pwd = st.secrets["Dm@302700"]
        to = st.secrets["alert_bot@gmail.com"]
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, pwd)
            smtp.send_message(msg)
    except:
        pass
    
# --------------- Dynamic Ticker Resolver (FMP API) ---------------
def resolve_ticker(user_input):
    api_key = st.secrets.get("fmp_api_key", "2GEUl972EPLW79v4I5mlg7s32GkG0Kk9")
    base_url = "https://financialmodelingprep.com/api/v3/search"
    params = {
        "query": user_input.strip(),
        "limit": 1,
        "exchange": "NASDAQ,NYSE,CRYPTO,COMMODITY",
        "apikey": api_key
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        if data and 'symbol' in data[0]:
            return data[0]['symbol']
        else:
            return user_input.upper()  # fallback if not found
    except Exception as e:
        st.warning(f"⚠️ Error resolving ticker for '{user_input}': {e}")
        return user_input.upper()

# --------------- UI ---------------
st.set_page_config(page_title="StarkAI v10", layout="centered")
st.title("🤖 StarkAI – GPT-Enabled Market Assistant")
st.caption("Real-time insights. Web scraping. Voice & GPT-powered trading assistant.")

trading_enabled = st.checkbox("🔒 Enable Simulated Trading", value=False)
tickers = st.text_input("📌 Enter Ticker(s) (e.g., AAPL, TSLA, BTC)", "")
# Dynamically resolve all tickers from user input
ticker_list = [resolve_ticker(t) for t in tickers.split(",") if t.strip()]


if st.button("🎤 Use Voice Input"):
    voice_result = recognize_speech()
    if voice_result:
        ticker_list = [voice_result]
        st.success(f"🎯 Recognized: {voice_result}")

if st.button("🧠 Analyze Market"):
    if not ticker_list:
        st.warning("Please enter or speak a ticker.")
    else:
        for ticker in ticker_list:
            st.header(f"📊 {ticker}")
            with st.spinner("Analyzing..."):

                # News
                try:
                    headlines = scrape_news(ticker)
                    st.subheader("🗞️ News & Sentiment")
                    for i, h in enumerate(headlines, 1):
                        sentiment = analyze_sentiment(h)
                        st.write(f"{i}. {h}")
                        st.caption(f"Sentiment: {sentiment}")
                except:
                    headlines = []
                    st.warning("⚠️ Failed to fetch news.")

                # Prediction
                st.subheader("📈 Price Prediction")
                prediction = predict_stock(ticker)
                if prediction:
                    st.write(prediction)
                else:
                    st.error("❌ No market data.")

                # Jarvis Recommendation
                st.subheader("💡 Jarvis Recommendation")
                recommendation = give_recommendation(prediction, ticker)
                st.success(recommendation)
                speak(recommendation)

                # GPT Commentary
                st.subheader("🧠 GPT-4 Market Commentary")
                gpt = gpt_commentary(ticker, headlines, prediction)
                st.info(gpt)

                # Simulated Trade
                st.subheader("🧪 Simulated Trade")
                trade = simulate_trade(ticker, recommendation, trading_enabled)
                st.code(trade)

                # Alerts
                send_telegram(f"StarkAI Alert ({ticker}):\n{recommendation}")
                send_email(f"StarkAI Update: {ticker}", recommendation)

                # Chart
                st.subheader("📉 Price Chart")
                try:
                    df = yf.download(ticker, period="5d", interval="1h")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=ticker))
                    fig.update_layout(title=f"{ticker} Price", xaxis_title="Time", yaxis_title="Price")
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.warning("⚠️ Chart unavailable.")
