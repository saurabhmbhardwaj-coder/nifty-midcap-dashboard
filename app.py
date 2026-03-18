import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from arch import arch_model
import time
import os

# ---------------- CONFIG ---------------- #

st.set_page_config(
page_title="Volatility Analytics Dashboard",
layout="wide"
)

# ---------------- THEME ---------------- #

st.markdown("""
<style>
.stApp {background-color:#2F3E2F;color:#F5F5DC;}
h1,h2,h3 {color:#F5F5DC;}
[data-testid="stMetric"] {
background-color:#3E533E;
padding:10px;border-radius:8px;
}
.stButton>button {
background-color:#556B2F;color:#F5F5DC;
}
</style>
""", unsafe_allow_html=True)

st.title("Volatility Analytics Dashboard")

# ---------------- LOAD COMPANY LIST ---------------- #

file_path="midcap150.xlsx"

if not os.path.exists(file_path):
    st.error("midcap150.xlsx file missing")
    st.stop()

companies=pd.read_excel(file_path)
companies["Ticker"]=companies["Symbol"]+".NS"

company=st.selectbox("Select Company",companies["Company Name"])

row=companies[companies["Company Name"]==company].iloc[0]

ticker=row["Ticker"]
industry=row["Industry"]

# ---------------- SAFE DATA LOADER ---------------- #

@st.cache_data(ttl=3600)
def load_data(ticker):

    try:
        data=yf.download(
            ticker,
            period="2y",
            progress=False,
            threads=False
        )

        if data.empty:
            return pd.DataFrame()

        if isinstance(data.columns,pd.MultiIndex):
            data.columns=data.columns.get_level_values(0)

        return data.reset_index()

    except:
        return pd.DataFrame()

data=load_data(ticker)

if data.empty:
    st.error("Data not available")
    st.stop()

current_price=float(data["Close"].iloc[-1])

# ---------------- GARCH ---------------- #

@st.cache_data
def compute_garch(returns):

    try:
        model=arch_model(returns,p=1,q=1)
        res=model.fit(disp="off")
        forecast=res.forecast(horizon=10)
        return np.sqrt(forecast.variance.iloc[-1,0]), np.sqrt(forecast.variance.values[-1])
    except:
        return np.nan, np.zeros(10)

returns=100*data["Close"].pct_change().dropna()

volatility, forecast_curve = compute_garch(returns)

# ---------------- KPI ---------------- #

c1,c2,c3,c4=st.columns(4)

c1.metric("Company",company)
c2.metric("Sector",industry)
c3.metric("Current Price",round(current_price,2))
c4.metric("Volatility",round(volatility,2))

# ---------------- TABS ---------------- #

tab1,tab2,tab3,tab4,tab5 = st.tabs([
"Stock Overview",
"Volatility Forecast",
"Risk-Return",
"Sector Dashboard",
"Portfolio"
])

# ---------------- PRICE ---------------- #

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

# ---------------- VOL FORECAST ---------------- #

with tab2:

    df=pd.DataFrame({
    "Step":range(1,11),
    "Volatility":forecast_curve
    })

    fig=go.Figure()

    fig.add_trace(go.Scatter(
    x=df["Step"],
    y=df["Volatility"],
    mode="lines+markers"
    ))

    fig.update_layout(
    xaxis_title="Forecast Horizon",
    yaxis_title="Volatility"
    )

    st.plotly_chart(fig,use_container_width=True)

    # MULTI FACTOR (BETA)

    market=load_data("^NSEI")

    if not market.empty:

        sr=data["Close"].pct_change().dropna()
        mr=market["Close"].pct_change().dropna()

        comb=pd.concat([sr,mr],axis=1).dropna()
        comb.columns=["Stock","Market"]

        beta=np.cov(comb["Stock"],comb["Market"])[0][1]/np.var(comb["Market"])

        st.metric("Market Beta",round(beta,3))

# ---------------- RISK RETURN ---------------- #

with tab3:

    stocks=st.multiselect("Select Stocks",companies["Company Name"])

    if stocks:

        rr=[]

        for s in stocks[:10]:

            time.sleep(0.2)

            t=companies.loc[companies["Company Name"]==s,"Ticker"].values[0]

            d=load_data(t)

            if d.empty: continue

            r=d["Close"].pct_change().dropna()

            ret=r.mean()*252
            risk=r.std()*np.sqrt(252)

            rr.append([s,ret,risk])

        df=pd.DataFrame(rr,columns=["Stock","Return","Risk"])

        fig=go.Figure()

        fig.add_trace(go.Scatter(
        x=df["Risk"],
        y=df["Return"],
        mode="markers+text",
        text=df["Stock"]
        ))

        fig.update_layout(
        xaxis_title="Risk",
        yaxis_title="Return"
        )

        st.plotly_chart(fig,use_container_width=True)

# ---------------- SECTOR ---------------- #

with tab4:

    sector_data=companies[companies["Industry"]==industry]

    names=[]
    vols=[]

    for _,r in sector_data.head(10).iterrows():

        time.sleep(0.2)

        d=load_data(r["Ticker"])

        if d.empty: continue

        ret=100*d["Close"].pct_change().dropna()

        v,_=compute_garch(ret)

        names.append(r["Company Name"])
        vols.append(v)

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

# ---------------- PORTFOLIO ---------------- #

with tab5:

    portfolio=st.multiselect("Select Portfolio",companies["Company Name"])

    if portfolio:

        returns_df=pd.DataFrame()

        for s in portfolio[:10]:

            t=companies.loc[companies["Company Name"]==s,"Ticker"].values[0]

            d=load_data(t)

            if d.empty: continue

            returns_df[s]=d["Close"].pct_change()

        returns_df=returns_df.dropna()

        if not returns_df.empty:

            w=np.ones(len(returns_df.columns))/len(returns_df.columns)

            cov=returns_df.cov()

            vol=np.sqrt(np.dot(w.T,np.dot(cov,w)))

            st.metric("Portfolio Volatility",round(vol,4))

            st.line_chart(returns_df)

# ---------------- SAFE FINANCIAL METRICS ---------------- #

def get_safe_metrics(ticker):

    try:
        info=yf.Ticker(ticker).fast_info

        return {
        "Last Price":info.get("lastPrice"),
        "Day High":info.get("dayHigh"),
        "Day Low":info.get("dayLow"),
        "Market Cap":info.get("marketCap")
        }

    except:
        return {}

st.subheader("Key Financial Metrics")

metrics=get_safe_metrics(ticker)

df=pd.DataFrame(metrics.items(),columns=["Metric","Value"])

st.table(df)

# ---------------- DOWNLOAD ---------------- #

st.download_button(
"Download Data CSV",
data.to_csv(),
file_name=f"{ticker}.csv"
)
