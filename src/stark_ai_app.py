
# === stark_ai_app.py (StarkAI v10.1 – FMP Integration Upgrade) ===

import streamlit as st
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import pyttsx3
import speech_recognition as sr
from openai import OpenAI
import plotly.graph_objects as go
import smtplib
import pandas as pd
from email.mime.text import MIMEText

# --------------- Secrets Setup ---------------
FMP_API_KEY = "2GEUl972EPLW79v4I5mlg7s32GkG0Kk9"
client = OpenAI(api_key=st.secrets.get("openai_api_key", ""))

# --------------- Normalize Ticker ---------------
def normalize_ticker(ticker):
    mapping = {
        "BTC": "BTCUSD",
        "ETH": "ETHUSD",
        "GOLD": "XAUUSD",
        "SILVER": "XAGUSD",
        "OIL": "WTIUSD",
        "USD": "USDEUR",
    }
    return mapping.get(ticker.upper(), ticker.upper())

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


