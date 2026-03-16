import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from arch import arch_model
import os

# ---------------- PAGE SETTINGS ---------------- #

st.set_page_config(
page_title="Volatility Analysis",
layout="wide",
initial_sidebar_state="collapsed"
)

# ---------------- CONSULTING STYLE THEME ---------------- #

st.markdown("""
<style>

.stApp{
background-color:#F5F5DC;
color:#2F3E2F;
}

h1,h2,h3{
color:#3A4F3A;
}

.stButton>button{
background-color:#556B2F;
color:white;
border-radius:8px;
}

.stMetric{
background-color:#E8E5D2;
padding:10px;
border-radius:10px;
}

</style>
""",unsafe_allow_html=True)

st.title("Volatility Analysis Dashboard")

# ---------------- LOAD COMPANY LIST ---------------- #

file_path="midcap150.xlsx"

if not os.path.exists(file_path):
    st.error("midcap150.xlsx file not found in repository")
    st.stop()

companies=pd.read_excel(file_path)

companies["Ticker"]=companies["Symbol"]+".NS"

company=st.selectbox("Select Company",companies["Company Name"])

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

# ---------------- TABS ---------------- #

tab1,tab2,tab3,tab4 = st.tabs([
"Stock Overview",
"Sector Analysis",
"Financial Ratios",
"Learn Financials"
])

# ================= STOCK TAB ================= #

with tab1:

    col1,col2=st.columns([3,1])

    with col1:

        fig=go.Figure()

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

        st.plotly_chart(fig,use_container_width=True)

    returns=100*data["Close"].pct_change().dropna()

    model=arch_model(returns,p=1,q=1)

    result=model.fit(disp="off")

    forecast=result.forecast(horizon=5)

    volatility=np.sqrt(forecast.variance.iloc[-1,0])

    with col2:

        st.metric("GARCH Volatility",round(volatility,2))

        if volatility<1.5:
            st.success("Low Volatility")

        elif volatility<3:
            st.warning("Moderate Volatility")

        else:
            st.error("High Volatility")

        st.write("### Volatility Interpretation")

        scale_df=pd.DataFrame({

        "Range":["0 - 1.5","1.5 - 3","3+"],

        "Meaning":[
        "Low Risk",
        "Moderate Risk",
        "High Risk"
        ]

        })

        st.table(scale_df)

# ================= SECTOR TAB ================= #

with tab2:

    st.subheader("Sector Volatility Analysis")

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
    yaxis_title="GARCH Volatility"
    )

    st.plotly_chart(fig3,use_container_width=True)

    sector_avg=np.mean(sector_vol)

    st.subheader("Sector Comparison")

    comparison_df=pd.DataFrame({

    "Metric":[
    "Selected Company",
    "Sector Average"
    ],

    "Volatility":[
    volatility,
    sector_avg
    ]

    })

    st.table(comparison_df)

# ================= FINANCIAL RATIOS ================= #

with tab3:

    st.subheader("Financial Ratios")

    info=yf.Ticker(ticker).info

    ratios={

    "PE Ratio":info.get("trailingPE"),

    "Price to Book":info.get("priceToBook"),

    "Return on Equity":info.get("returnOnEquity"),

    "Debt to Equity":info.get("debtToEquity"),

    "Profit Margin":info.get("profitMargins")

    }

    ratio_df=pd.DataFrame(
    ratios.items(),
    columns=["Ratio","Value"]
    )

    st.table(ratio_df)

    st.download_button(
    "Download CSV",
    data.to_csv(),
    file_name=f"{ticker}_data.csv"
    )

# ================= LEARN TAB ================= #

with tab4:

    st.subheader("Learn Financial Ratios")

    learn_df=pd.DataFrame({

    "Ratio":[
    "PE Ratio",
    "Price to Book",
    "Return on Equity",
    "Debt to Equity",
    "Profit Margin"
    ],

    "Definition":[
    "Market price divided by earnings per share",
    "Market value compared to book value",
    "Net income divided by shareholder equity",
    "Total debt divided by equity",
    "Net profit divided by revenue"
    ],

    "Interpretation":[
    "Higher PE may indicate growth expectations",
    "High value means market values company above assets",
    "Higher ROE indicates efficient capital usage",
    "Higher ratio means more financial leverage",
    "Higher margin indicates better profitability"
    ]

    })

    st.table(learn_df)
