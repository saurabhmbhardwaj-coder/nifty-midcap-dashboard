import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_loader import load_data
from volatility import garch_volatility
from ratios import get_ratios
from stocks import stocks

st.set_page_config(layout="wide")

st.title("Volatility Analysis Dashboard")

# -------- STOCK LIST -------- #

all_tickers = []

for sector in stocks:
    all_tickers += stocks[sector]

selected = st.selectbox("Select Stock", all_tickers)

# -------- LOAD DATA -------- #

data = load_data(selected)

if data is None or data.empty:

    st.error("Unable to load stock data.")

else:

    # ---- FIX MULTI INDEX COLUMNS ---- #

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # ---- RESET INDEX ---- #

    data = data.reset_index()

    if "Close" not in data.columns:

        st.error("Close price column missing.")

    else:

        col1, col2 = st.columns([3,1])

        with col1:

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data["Close"],
                    mode="lines",
                    name="Price"
                )
            )

            fig.update_layout(
                title="Stock Price (Last 2 Years)",
                xaxis_title="Date",
                yaxis_title="Price"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:

            vol = garch_volatility(data["Close"])

            st.metric("GARCH Volatility", round(vol,2))

            if vol < 1.5:
                st.success("Low Volatility")

            elif vol < 3:
                st.warning("Moderate Volatility")

            else:
                st.error("High Volatility")


# -------- INDUSTRY COMPARISON -------- #

st.subheader("Industry Volatility Comparison")

sector_name = None

for s in stocks:
    if selected in stocks[s]:
        sector_name = s

comparison = []

if sector_name:

    for ticker in stocks[sector_name]:

        d = load_data(ticker)

        if d is not None and not d.empty:

            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)

            vol = garch_volatility(d["Close"])

            comparison.append([ticker, vol])

if comparison:

    df = pd.DataFrame(comparison, columns=["Stock","Volatility"])

    st.bar_chart(df.set_index("Stock"))


# -------- CSV EXPORT -------- #

st.download_button(
"Download CSV",
data.to_csv(),
"stock_data.csv"
)


# -------- SECTOR VOLATILITY BUTTONS -------- #

st.subheader("Sector Volatility")

cols = st.columns(len(stocks))

for i,sector in enumerate(stocks):

    if cols[i].button(sector):

        vols = []

        for ticker in stocks[sector]:

            d = load_data(ticker)

            if d is not None and not d.empty:

                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)

                vols.append(garch_volatility(d["Close"]))

        sector_df = pd.DataFrame(vols, columns=["Volatility"])

        st.line_chart(sector_df)


# -------- FINANCIAL RATIOS -------- #

st.subheader("Financial Ratios")

ratios = get_ratios(selected)

st.json(ratios)
