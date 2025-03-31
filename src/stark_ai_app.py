# === stark_ai_app.py (Smart Version – No URL Input) ===

import streamlit as st
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from textblob import TextBlob
import pyttsx3

# ------------------ Jarvis Voice ------------------
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.warning(f"🗣️ Voice error: {e}")

# ------------------ Sentiment Analyzer ------------------
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "🟢 Positive"
    elif polarity < -0.2:
        return "🔴 Negative"
    else:
        return "🟡 Neutral"

# ------------------ News Scraper ------------------
def scrape_news(ticker):
    url = f"https://news.google.com/search?q={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("a", class_="DY5T1d", limit=5)
    return [article.text for article in articles]

# ------------------ Real-Time Stock Predictor ------------------
def predict_stock(ticker):
    data = yf.download(ticker, period="5d", interval="1h")
    if data.empty:
        return None
    latest_close = data['Close'].iloc[-1]
    avg_close = data['Close'].mean()
    volume = data['Volume'].iloc[-1]
    trend = "Up" if latest_close > avg_close else "Down"
    return {
        "latest_close": round(latest_close, 2),
        "average_close": round(avg_close, 2),
        "volume": int(volume),
        "trend": trend
    }

# ------------------ Recommendation Engine ------------------
def give_recommendation(prediction, ticker):
    if not prediction:
        return "❌ No data available."
    if prediction["trend"] == "Up" and prediction["volume"] > 1000000:
        return f"🚀 Strong Buy for {ticker}! Price is rising with high volume."
    elif prediction["trend"] == "Up":
        return f"📈 Mild Buy for {ticker} — Uptrend detected, but low volume."
    elif prediction["trend"] == "Down":
        return f"⚠️ Caution: {ticker} is trending down. Consider holding."
    else:
        return f"❓ Not enough info for a confident signal."

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="StarkAI", layout="centered")

st.title("🤖 StarkAI – Autonomous AI Market Assistant")
st.caption("Real-time prediction. Web-scraping intelligence. Jarvis voice included.")

ticker = st.text_input("📌 Enter Ticker Symbol (e.g., AAPL, TSLA, BTC, GOLD)", "")

if st.button("Analyze"):
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        with st.spinner("🧠 Scraping intelligence and analyzing market..."):

            # Scrape & analyze news
            st.subheader("🗞️ Latest News Headlines & Sentiment")
            try:
                headlines = scrape_news(ticker)
                for i, headline in enumerate(headlines, 1):
                    sentiment = analyze_sentiment(headline)
                    st.write(f"{i}. {headline}")
                    st.caption(f"Sentiment: {sentiment}")
            except:
                st.error("⚠️ News scraping failed.")

            # Predict market trend
            st.subheader("📈 Real-Time Price Prediction")
            prediction = predict_stock(ticker)
            if prediction:
                st.write(f"Latest Close: ${prediction['latest_close']}")
                st.write(f"5-Day Avg: ${prediction['average_close']}")
                st.write(f"Volume: {prediction['volume']}")
                st.write(f"Trend: **{prediction['trend']}**")
            else:
                st.warning("⚠️ Could not retrieve stock data.")

            # Final recommendation
            st.subheader("💡 Jarvis Recommendation")
            recommendation = give_recommendation(prediction, ticker)
            st.success(recommendation)
            speak(recommendation)
