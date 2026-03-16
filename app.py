import streamlit as st
import yfinance as yf

st.title("Nifty Midcap Dashboard")

ticker = st.text_input("Enter Stock Ticker (example: PERSISTENT.NS)")

if ticker:

    data = yf.download(ticker, period="1y")

    st.subheader("Stock Price Chart")

    st.line_chart(data["Adj Close"])

    st.subheader("Raw Data")

    st.write(data)
