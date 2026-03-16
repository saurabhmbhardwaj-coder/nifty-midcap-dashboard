import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from arch import arch_model
import os

# ---------------- PAGE SETTINGS ---------------- #

st.set_page_config(
page_title="Volatility Analysis Dashboard",
layout="wide",
initial_sidebar_state="collapsed"
)

# ---------------- DARK OLIVE THEME ---------------- #

st.markdown("""
<style>

.stApp{
background-color:#2F3E2F;
color:#F5F5DC;
}

h1,h2,h3{
color:#F5F5DC;
}

[data-testid="stMetric"]{
background-color:#3E533E;
padding:10px;
border-radius:8px;
color:#F5F5DC;
}

.stButton>button{
background-color:#556B2F;
color:#F5F5DC;
border-radius:8px;
}

</style>
""",unsafe_allow_html=True)

st.title("Volatility Analysis Dashboard")

# ---------------- LOAD COMPANY LIST ---------------- #

file_path="midcap150.xlsx"

if not os.path.exists(file_path):
    st.error("midcap150.xlsx file not found.")
    st.stop()

companies=pd.read_excel(file_path)

companies["Ticker"]=companies["Symbol"]+".NS"

company=st.selectbox(
"Select Company",
companies["Company Name"],
index=None
)

if company is None:
    st.stop()

row=companies[companies["Company Name"]==company].iloc[0]

ticker=row["Ticker"]
industry=row["Industry"]

# ---------------- DATA LOADER ---------------- #

@st.cache_data
def load_data(ticker):

    data=yf.download(ticker,period="2y",progress=False)

    if isinstance(data.columns,pd.MultiIndex):
        data.columns=data.columns.get_level_values(0)

    data=data.reset_index()

    return data

data=load_data(ticker)

if data.empty:
    st.error("Unable to fetch stock data")
    st.stop()

# ---------------- CURRENT PRICE ---------------- #

current_price=float(data["Close"].iloc[-1])

# ---------------- GARCH MODEL ---------------- #

returns=100*data["Close"].pct_change().dropna()

model=arch_model(returns,p=1,q=1)

result=model.fit(disp="off")

forecast=result.forecast(horizon=10)

volatility=np.sqrt(forecast.variance.iloc[-1,0])

forecast_curve=np.sqrt(forecast.variance.values[-1])

# ---------------- KPI STRIP ---------------- #

col1,col2,col3,col4,col5=st.columns(5)

with col1:
    st.metric("Company",company)

with col2:
    st.metric("Industry",industry)

with col3:
    st.metric("Current Market Price",round(current_price,2))

with col4:
    st.metric("GARCH Volatility",round(volatility,2))

with col5:

    if volatility<1.5:
        risk="Low"
    elif volatility<3:
        risk="Moderate"
    else:
        risk="High"

    st.metric("Risk Level",risk)

# ---------------- TABS ---------------- #

tab1,tab2,tab3,tab4 = st.tabs([
"Stock Overview",
"Volatility Forecast",
"Sector Analysis",
"Learn Financials"
])

# ================= STOCK OVERVIEW ================= #

with tab1:

    fig=go.Figure()

    fig.add_trace(go.Scatter(
    x=data["Date"],
    y=data["Close"],
    mode="lines",
    name="Price"
    ))

    fig.update_layout(
    title="Stock Price (Last 2 Years)",
    xaxis_title="Date",
    yaxis_title="Price"
    )

    st.plotly_chart(fig,use_container_width=True)

# ================= VOLATILITY FORECAST ================= #

with tab2:

    st.subheader("GARCH Volatility Forecast Curve")

    forecast_df=pd.DataFrame({
    "Forecast Period":range(1,11),
    "Volatility":forecast_curve
    })

    fig2=go.Figure()

    fig2.add_trace(go.Scatter(
    x=forecast_df["Forecast Period"],
    y=forecast_df["Volatility"],
    mode="lines+markers"
    ))

    fig2.update_layout(
    xaxis_title="Forecast Horizon",
    yaxis_title="Predicted Volatility"
    )

    st.plotly_chart(fig2,use_container_width=True)

    scale_df=pd.DataFrame({

    "Range":["0-1.5","1.5-3","3+"],

    "Interpretation":[
    "Low Volatility",
    "Moderate Volatility",
    "High Volatility"
    ]

    })

    st.table(scale_df)

# ================= SECTOR ANALYSIS ================= #

with tab3:

    sector_data=companies[companies["Industry"]==industry]

    sector_names=[]
    sector_vol=[]

    for _,r in sector_data.iterrows():

        try:

            d=load_data(r["Ticker"])

            ret=100*d["Close"].pct_change().dropna()

            m=arch_model(ret,p=1,q=1)

            res=m.fit(disp="off")

            f=res.forecast(horizon=5)

            v=np.sqrt(f.variance.iloc[-1,0])

            sector_vol.append(v)

            sector_names.append(r["Company Name"])

        except:
            pass

    fig3=go.Figure()

    fig3.add_trace(go.Scatter(
    x=sector_names,
    y=sector_vol,
    mode="lines+markers"
    ))

    fig3.update_layout(
    title=f"{industry} Volatility Comparison",
    xaxis_title="Companies",
    yaxis_title="Volatility"
    )

    st.plotly_chart(fig3,use_container_width=True)

# ================= LEARN FINANCIALS ================= #

with tab4:

    learn_df=pd.DataFrame({

    "Ratio":[
    "Price-Earnings Ratio",
    "Price-to-Book Ratio",
    "Return on Equity",
    "Debt-to-Equity Ratio",
    "Profit Margin",
    "Current Ratio",
    "Quick Ratio",
    "Interest Coverage Ratio"
    ],

    "Definition":[
    "Market price divided by earnings per share",
    "Market value relative to book value",
    "Net income divided by shareholder equity",
    "Total debt divided by equity",
    "Net profit divided by revenue",
    "Current assets divided by current liabilities",
    "Liquid assets divided by current liabilities",
    "EBIT divided by interest expense"
    ],

    "Interpretation":[
    "Higher PE may indicate growth expectations",
    "High PB suggests market values company above assets",
    "Higher ROE means efficient capital use",
    "High ratio indicates leverage risk",
    "Higher margin means stronger profitability",
    "Measures short-term liquidity",
    "Stricter liquidity indicator",
    "Shows ability to pay interest obligations"
    ]

    })

    st.table(learn_df)
