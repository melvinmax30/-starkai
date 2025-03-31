# === stark_ai_app.py (StarkAI v10.0 Quantum Upgrade - Final Fix - Corrected) ===

import streamlit as st
import requests
from bs4 import BeautifulSoup
from alpha_vantage.timeseries import TimeSeries
from textblob import TextBlob
import pyttsx3
import speech_recognition as sr
import openai
import plotly.graph_objects as go
import smtplib
import pandas as pd
from email.mime.text import MIMEText

# --------------- GPT Setup ---------------
client = openai.OpenAI(api_key=st.secrets["openai_api_key"])
ALPHA_VANTAGE_API_KEY = 'VONTVIEU6DDNM23E'
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

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
    try:
        data, meta_data = ts.get_intraday(symbol=ticker, interval='60min', outputsize='compact')

        if data.empty:
            st.error(f"❌ No market data returned for {ticker}. Check symbol or API limits.")
            return None

        data = data.sort_index()  # Ensure data is sorted by time (ascending)
        
        data.columns = ["Open", "High", "Low", "Close", "Volume"]
        
        latest_close = data["Close"].iloc[-1]
        avg_close = data["Close"].mean()
        volume = data["Volume"].iloc[-1]

        trend = "Up" if latest_close > avg_close else "Down"

        return {
            "latest_close": round(latest_close, 2),
            "average_close": round(avg_close, 2),
            "volume": int(volume),
            "trend": trend
        }

    except Exception as e:
        st.error(f"⚠️ Alpha Vantage API error: {e}")
        return None

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

    Provide a concise market commentary and clear recommendation in plain English.
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5 turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return completion.choices[0].message.content.strip()
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
        token = st.secrets["telegram_token"]
        chat_id = st.secrets["telegram_chat_id"]
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": msg})
    except:
        pass

# --------------- Email Alert ---------------
def send_email(subject, body):
    try:
        sender = st.secrets["email_sender"]
        pwd = st.secrets["email_password"]
        to = st.secrets["email_recipient"]
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, pwd)
            smtp.send_message(msg)
    except:
        pass

# --------------- UI ---------------
st.set_page_config(page_title="StarkAI v10", layout="centered")
st.title("🤖 StarkAI – GPT-Enabled Market Assistant")
st.caption("Real-time insights. Web scraping. Voice & GPT-powered trading assistant.")

trading_enabled = st.checkbox("🔒 Enable Simulated Trading", value=False)
tickers = st.text_input("📌 Enter Ticker(s) (e.g., AAPL, TSLA, BTC)", "")
ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

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

                st.subheader("📈 Price Prediction")
                prediction = predict_stock(ticker)
                if prediction:
                    st.write(prediction)
                else:
                    st.error("❌ No market data.")

                st.subheader("💡 Jarvis Recommendation")
                recommendation = give_recommendation(prediction, ticker)
                st.success(recommendation)
                speak(recommendation)

                st.subheader("🧠 GPT-4 Market Commentary")
                gpt = gpt_commentary(ticker, headlines, prediction)
                st.info(gpt)

                st.subheader("🧪 Simulated Trade")
                trade = simulate_trade(ticker, recommendation, trading_enabled)
                st.code(trade)

                send_telegram(f"StarkAI Alert ({ticker}): {recommendation}")

                send_email(f"StarkAI Update: {ticker}", recommendation)

                st.subheader("📉 Price Chart")
try:
    df, meta_data = ts.get_intraday(symbol=ticker, interval='60min', outputsize='compact')
    df = df.sort_index()
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=ticker))
    fig.update_layout(title=f"{ticker} Price", xaxis_title="Time", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"⚠️ Chart unavailable: {e}")
