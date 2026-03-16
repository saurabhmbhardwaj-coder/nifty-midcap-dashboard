import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from arch import arch_model
import os

st.set_page_config(
page_title="Volatility Analysis Dashboard",
layout="wide",
initial_sidebar_state="collapsed"
)

# ---------------- THEME ---------------- #

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

# ---------------- COMPANY LIST ---------------- #

file_path="midcap150.xlsx"

if not os.path.exists(file_path):
    st.error("midcap150.xlsx file not found.")
    st.stop()

companies=pd.read_excel(file_path)

companies["Ticker"]=companies["Symbol"]+".NS"

company=st.selectbox("Select Company",companies["Company Name"])

row=companies[companies["Company Name"]==company].iloc[0]

ticker=row["Ticker"]
industry=row["Industry"]

# ---------------- DATA ---------------- #

@st.cache_data
def load_data(t):

    d=yf.download(t,period="2y",progress=False)

    if isinstance(d.columns,pd.MultiIndex):
        d.columns=d.columns.get_level_values(0)

    d=d.reset_index()

    return d

data=load_data(ticker)

if data.empty:
    st.error("Stock data unavailable")
    st.stop()

current_price=float(data["Close"].iloc[-1])

# ---------------- GARCH ---------------- #

returns=100*data["Close"].pct_change().dropna()

model=arch_model(returns,p=1,q=1)

result=model.fit(disp="off")

forecast=result.forecast(horizon=10)

volatility=np.sqrt(forecast.variance.iloc[-1,0])

forecast_curve=np.sqrt(forecast.variance.values[-1])

# ---------------- KPI STRIP ---------------- #

col1,col2,col3,col4=st.columns(4)

col1.metric("Company",company)
col2.metric("Sector",industry)
col3.metric("Current Price",round(current_price,2))
col4.metric("GARCH Volatility",round(volatility,2))

# ---------------- TABS ---------------- #

tab1,tab2,tab3,tab4,tab5 = st.tabs([
"Stock Overview",
"Volatility Forecast",
"Sector Analysis",
"Portfolio Analytics",
"Financials"
])

# ---------------- PRICE CHART ---------------- #

with tab1:

    fig=go.Figure()

    fig.add_trace(go.Scatter(
    x=data["Date"],
    y=data["Close"],
    mode="lines"
    ))

    fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price"
    )

    st.plotly_chart(fig,use_container_width=True)

# ---------------- VOLATILITY FORECAST ---------------- #

with tab2:

    forecast_df=pd.DataFrame({
    "Forecast Period":range(1,11),
    "Volatility":forecast_curve
    })

    fig=go.Figure()

    fig.add_trace(go.Scatter(
    x=forecast_df["Forecast Period"],
    y=forecast_df["Volatility"],
    mode="lines+markers"
    ))

    fig.update_layout(
    xaxis_title="Forecast Horizon",
    yaxis_title="Volatility"
    )

    st.plotly_chart(fig,use_container_width=True)

# ---------------- SECTOR ANALYSIS ---------------- #

with tab3:

    sector_data=companies[companies["Industry"]==industry]

    names=[]
    vols=[]

    for _,r in sector_data.iterrows():

        try:

            d=load_data(r["Ticker"])

            ret=100*d["Close"].pct_change().dropna()

            m=arch_model(ret,p=1,q=1)

            res=m.fit(disp="off")

            f=res.forecast(horizon=5)

            v=np.sqrt(f.variance.iloc[-1,0])

            names.append(r["Company Name"])
            vols.append(v)

        except:
            pass

    fig=go.Figure()

    fig.add_trace(go.Scatter(
    x=names,
    y=vols,
    mode="lines+markers"
    ))

    fig.update_layout(
    xaxis_title="Companies",
    yaxis_title="Volatility"
    )

    st.plotly_chart(fig,use_container_width=True)

# ---------------- PORTFOLIO ANALYTICS ---------------- #

with tab4:

    portfolio=st.multiselect(
    "Select Portfolio Stocks",
    companies["Company Name"]
    )

    if portfolio:

        returns_df=pd.DataFrame()

        for stock in portfolio:

            t=companies.loc[
            companies["Company Name"]==stock,"Ticker"
            ].values[0]

            d=load_data(t)

            returns_df[stock]=d["Close"].pct_change()

        returns_df=returns_df.dropna()

        weights=np.ones(len(portfolio))/len(portfolio)

        cov=returns_df.cov()

        port_vol=np.sqrt(
        np.dot(weights.T,np.dot(cov,weights))
        )

        st.metric("Portfolio Volatility",round(port_vol,4))

        st.line_chart(returns_df)

# ---------------- FINANCIAL RATIOS ---------------- #

with tab5:

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

    st.subheader("Ratio Interpretation")

    learn_df=pd.DataFrame({

    "Ratio":[
    "PE Ratio",
    "Price to Book",
    "Return on Equity",
    "Debt to Equity",
    "Profit Margin"
    ],

    "Interpretation":[
    "Shows valuation relative to earnings",
    "Compares market value with assets",
    "Measures efficiency of shareholder capital",
    "Indicates financial leverage",
    "Shows company profitability"
    ]

    })

    st.table(learn_df)

st.download_button(
"Download CSV",
data.to_csv(),
file_name=f"{ticker}_data.csv"
)
