import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from arch import arch_model

st.set_page_config(layout="wide")

st.title("Volatility Analysis Dashboard")

# ---------------- LOAD COMPANY LIST ---------------- #

@st.cache_data
def load_companies():

    df = pd.read_excel("midcap150.xlsx")

    df["Ticker"] = df["Symbol"] + ".NS"

    return df

companies = load_companies()

# dropdown
company = st.selectbox(
"Select Company",
companies["Company Name"]
)

row = companies[companies["Company Name"] == company].iloc[0]

ticker = row["Ticker"]
industry = row["Industry"]

# ---------------- LOAD STOCK DATA ---------------- #

@st.cache_data
def load_data(ticker):

    data = yf.download(ticker, period="2y", progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    return data

data = load_data(ticker)

if data.empty:

    st.error("Unable to fetch stock data")

else:

    # ---------------- PRICE CHART ---------------- #

    col1, col2 = st.columns([3,1])

    with col1:

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data["Date"],
            y=data["Close"],
            mode="lines",
            name="Price"
        ))

        fig.update_layout(
        title=f"{company} Price Chart (2 Years)",
        xaxis_title="Date",
        yaxis_title="Price"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------- GARCH VOLATILITY ---------------- #

    returns = 100 * data["Close"].pct_change().dropna()

    model = arch_model(returns, p=1, q=1)

    result = model.fit(disp="off")

    forecast = result.forecast(horizon=5)

    volatility = np.sqrt(forecast.variance.iloc[-1,0])

    with col2:

        st.metric("GARCH Volatility", round(volatility,2))

        if volatility < 1.5:

            st.success("Low Volatility")

        elif volatility < 3:

            st.warning("Moderate Volatility")

        else:

            st.error("High Volatility")

# ---------------- INDUSTRY COMPARISON ---------------- #

st.subheader("Industry Comparison")

industry_companies = companies[companies["Industry"] == industry]

vol_list = []

for _, r in industry_companies.iterrows():

    try:

        d = load_data(r["Ticker"])

        ret = 100 * d["Close"].pct_change().dropna()

        m = arch_model(ret, p=1, q=1)

        res = m.fit(disp="off")

        f = res.forecast(horizon=5)

        v = np.sqrt(f.variance.iloc[-1,0])

        vol_list.append([r["Symbol"], v])

    except:

        pass

comp_df = pd.DataFrame(vol_list, columns=["Stock","Volatility"])

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=comp_df["Stock"],
    y=comp_df["Volatility"],
    mode="lines+markers"
))

fig2.update_layout(
title="Industry Volatility Comparison",
xaxis_title="Stock",
yaxis_title="Volatility"
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------- FINANCIAL RATIOS ---------------- #

st.subheader("Financial Ratios")

info = yf.Ticker(ticker).info

ratios = {

"PE Ratio": info.get("trailingPE"),
"Price to Book": info.get("priceToBook"),
"Return on Equity": info.get("returnOnEquity"),
"Debt to Equity": info.get("debtToEquity"),
"Profit Margin": info.get("profitMargins")

}

ratio_df = pd.DataFrame(
ratios.items(),
columns=["Ratio","Value"]
)

st.table(ratio_df)

# ---------------- SECTOR VOLATILITY ---------------- #

st.subheader("Sector Volatility Analysis")

sector_groups = companies.groupby("Industry")

selected_sector = st.selectbox(
"Select Industry",
companies["Industry"].unique()
)

sector_data = sector_groups.get_group(selected_sector)

sector_vol = []

for _, r in sector_data.iterrows():

    try:

        d = load_data(r["Ticker"])

        ret = 100 * d["Close"].pct_change().dropna()

        m = arch_model(ret, p=1, q=1)

        res = m.fit(disp="off")

        f = res.forecast(horizon=5)

        v = np.sqrt(f.variance.iloc[-1,0])

        sector_vol.append(v)

    except:

        pass

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    y=sector_vol,
    mode="lines+markers"
))

fig3.update_layout(
title=f"{selected_sector} Volatility Trend",
xaxis_title="Companies",
yaxis_title="Volatility"
)

st.plotly_chart(fig3)

# ---------------- CSV EXPORT ---------------- #

st.download_button(
"Download Stock Data",
data.to_csv(),
file_name=f"{ticker}_data.csv"
)
