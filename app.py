import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from arch import arch_model
import os

st.set_page_config(
page_title="Volatility Analysis",
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

st.title("Volatility Analysis")

# ---------------- LOAD COMPANY LIST ---------------- #

file_path="midcap150.xlsx"

if not os.path.exists(file_path):
    st.error("midcap150.xlsx file missing.")
    st.stop()

companies=pd.read_excel(file_path)

companies["Ticker"]=companies["Symbol"]+".NS"

company=st.selectbox(
"Select Company",
companies["Company Name"]
)

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
    st.error("Stock data unavailable.")
    st.stop()

current_price=float(data["Close"].iloc[-1])

# ---------------- GARCH VOLATILITY ---------------- #

returns=100*data["Close"].pct_change().dropna()

model=arch_model(returns,p=1,q=1)

res=model.fit(disp="off")

forecast=res.forecast(horizon=10)

volatility=np.sqrt(forecast.variance.iloc[-1,0])

forecast_curve=np.sqrt(forecast.variance.values[-1])

# ---------------- KPI STRIP ---------------- #

c1,c2,c3,c4=st.columns(4)

c1.metric("Company",company)
c2.metric("Sector",industry)
c3.metric("Current Price",round(current_price,2))
c4.metric("GARCH Volatility",round(volatility,2))

# ---------------- TABS ---------------- #

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
"Stock Overview",
"Volatility Forecast",
"Risk Return",
"Sector Dashboard",
"Portfolio Analytics",
"Financial Ratios"
])

# ---------------- STOCK CHART ---------------- #

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

    # Multi-factor volatility using market beta

    market_data=load_data("^NSEI")

    stock_returns=data["Close"].pct_change().dropna()
    market_returns=market_data["Close"].pct_change().dropna()

    combined=pd.concat([stock_returns,market_returns],axis=1).dropna()

    combined.columns=["Stock","Market"]

    beta=np.cov(combined["Stock"],combined["Market"])[0][1]/np.var(combined["Market"])

    st.metric("Market Beta (Factor Exposure)",round(beta,3))

# ---------------- RISK RETURN SCATTER ---------------- #

with tab3:

    stocks=st.multiselect(
    "Select Stocks",
    companies["Company Name"]
    )

    if stocks:

        rr_data=[]

        for s in stocks:

            t=companies.loc[
            companies["Company Name"]==s,"Ticker"
            ].values[0]

            d=load_data(t)

            r=d["Close"].pct_change().dropna()

            mean=r.mean()*252
            risk=r.std()*np.sqrt(252)

            rr_data.append([s,mean,risk])

        rr_df=pd.DataFrame(
        rr_data,
        columns=["Stock","Return","Risk"]
        )

        fig=go.Figure()

        fig.add_trace(go.Scatter(
        x=rr_df["Risk"],
        y=rr_df["Return"],
        mode="markers+text",
        text=rr_df["Stock"]
        ))

        fig.update_layout(
        xaxis_title="Risk (Volatility)",
        yaxis_title="Expected Return"
        )

        st.plotly_chart(fig,use_container_width=True)

# ---------------- INTERACTIVE SECTOR DASHBOARD ---------------- #

with tab4:

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

with tab5:

    portfolio=st.multiselect(
    "Select Portfolio Stocks",
    companies["Company Name"]
    )

    if portfolio:

        returns_df=pd.DataFrame()

        for s in portfolio:

            t=companies.loc[
            companies["Company Name"]==s,"Ticker"
            ].values[0]

            d=load_data(t)

            returns_df[s]=d["Close"].pct_change()

        returns_df=returns_df.dropna()

        weights=np.ones(len(portfolio))/len(portfolio)

        cov=returns_df.cov()

        port_vol=np.sqrt(
        np.dot(weights.T,np.dot(cov,weights))
        )

        st.metric("Portfolio Volatility",round(port_vol,4))

        st.line_chart(returns_df)

# ---------------- FINANCIAL RATIOS ---------------- #

with tab6:

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
