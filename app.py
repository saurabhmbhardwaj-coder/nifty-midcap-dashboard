import streamlit as st
import pandas as pd
import plotly.express as px

from data_loader import load_data
from volatility import garch_volatility
from ratios import get_ratios
from stocks import stocks
from heatmap import create_heatmap

st.set_page_config(layout="wide")

st.title("Quant Equity Research Dashboard")

# -------- STOCK LIST -------- #

all_tickers = []

for sector in stocks:
    all_tickers += stocks[sector]

selected = st.selectbox(
"Select Stock",
all_tickers
)

data = load_data(selected)

if not data.empty:

    col1,col2 = st.columns([3,1])

    with col1:

        fig = px.line(
        data,
        y="Close",
        title="Price Chart (2 Years)"
        )

        st.plotly_chart(fig,use_container_width=True)

    with col2:

        vol = garch_volatility(data["Close"])

        st.metric(
        "GARCH Volatility",
        round(vol,2)
        )

        if vol < 1.5:
            st.success("Low Volatility")

        elif vol < 3:
            st.warning("Moderate Volatility")

        else:
            st.error("High Volatility")

# -------- INDUSTRY COMPARISON -------- #

st.subheader("Industry Comparison")

sector_name=None

for s in stocks:

    if selected in stocks[s]:
        sector_name=s

comparison=[]

for ticker in stocks[sector_name]:

    d=load_data(ticker)

    if not d.empty:

        v=garch_volatility(d["Close"])

        comparison.append([ticker,v])

df=pd.DataFrame(
comparison,
columns=["Stock","Volatility"]
)

st.bar_chart(df.set_index("Stock"))

# -------- EXPORT -------- #

st.download_button(
"Download CSV",
data.to_csv(),
"stock_data.csv"
)

# -------- SECTOR BUTTONS -------- #

st.subheader("Sector Volatility")

cols=st.columns(len(stocks))

for i,sector in enumerate(stocks):

    if cols[i].button(sector):

        vols=[]

        for ticker in stocks[sector]:

            d=load_data(ticker)

            if not d.empty:

                vols.append(
                garch_volatility(d["Close"])
                )

        st.line_chart(vols)

# -------- RATIOS -------- #

st.subheader("Financial Ratios")

ratios=get_ratios(selected)

st.json(ratios)
