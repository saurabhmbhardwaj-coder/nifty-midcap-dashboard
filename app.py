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

# ---------------- THEME: DARK OLIVE BG + BEIGE TEXT ---------------- #

st.markdown("""
<style>
.stApp{
    background-color:#2F3E2F;
    color:#F5F5DC;
}
h1,h2,h3,h4{
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
    border:none;
}
table{
    color:#F5F5DC;
}
</style>
""", unsafe_allow_html=True)

st.title("Volatility Analysis Dashboard")

# ---------------- LOAD COMPANY LIST ---------------- #

file_path = "midcap150.xlsx"

if not os.path.exists(file_path):
    st.error("midcap150.xlsx file not found in repository.")
    st.stop()

companies = pd.read_excel(file_path)

required_cols = {"Company Name", "Industry", "Symbol"}
if not required_cols.issubset(set(companies.columns)):
    st.error("Excel must contain columns: Company Name, Industry, Symbol")
    st.stop()

companies["Ticker"] = companies["Symbol"].astype(str) + ".NS"

# Dropdown only (no text search field)
company = st.selectbox("Select Company", companies["Company Name"])

row = companies.loc[companies["Company Name"] == company].iloc[0]
ticker = row["Ticker"]
industry = row["Industry"]

# ---------------- DATA LOADER ---------------- #

@st.cache_data
def load_data(tkr):
    data = yf.download(tkr, period="2y", progress=False)
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    return data

data = load_data(ticker)

if data.empty or "Close" not in data.columns:
    st.error("Unable to fetch stock data.")
    st.stop()

# ---------------- CURRENT PRICE ---------------- #

current_price = float(data["Close"].iloc[-1])

# ---------------- GARCH VOLATILITY ---------------- #

returns = 100 * data["Close"].pct_change().dropna()

try:
    model = arch_model(returns, p=1, q=1)
    result = model.fit(disp="off")
    forecast = result.forecast(horizon=10)
    volatility = float(np.sqrt(forecast.variance.iloc[-1, 0]))
    forecast_curve = np.sqrt(forecast.variance.values[-1])
except Exception:
    volatility = np.nan
    forecast_curve = np.array([np.nan]*10)

# ---------------- KPI STRIP ---------------- #

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Company", company)

with col2:
    st.metric("Industry", industry)

with col3:
    st.metric("Current Market Price", round(current_price, 2))

with col4:
    st.metric("GARCH Volatility", None if np.isnan(volatility) else round(volatility, 2))

with col5:
    if np.isnan(volatility):
        risk = "N/A"
    elif volatility < 1.5:
        risk = "Low"
    elif volatility < 3:
        risk = "Moderate"
    else:
        risk = "High"
    st.metric("Risk Level", risk)

# ---------------- TABS ---------------- #

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Stock Overview",
    "Volatility Forecast",
    "Sector Analysis",
    "Sector Heatmap",
    "Portfolio Analytics",
    "Factor Exposure",
    "Financials & Learn"
])

# ================= STOCK OVERVIEW ================= #

with tab1:
    st.subheader("Stock Price (Last 2 Years)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["Date"],
        y=data["Close"],
        mode="lines",
        name="Price"
    ))
    fig.update_layout(xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

# ================= VOLATILITY FORECAST ================= #

with tab2:
    st.subheader("GARCH Volatility Forecast Curve")

    forecast_df = pd.DataFrame({
        "Forecast Period": range(1, len(forecast_curve) + 1),
        "Volatility": forecast_curve
    })

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=forecast_df["Forecast Period"],
        y=forecast_df["Volatility"],
        mode="lines+markers"
    ))
    fig2.update_layout(
        xaxis_title="Forecast Horizon",
        yaxis_title="Predicted Volatility"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Multi-Stock Volatility Forecast")

    multi = st.multiselect("Select Stocks", companies["Company Name"])

    if multi:
        results = []
        for stock in multi:
            tkr = companies.loc[companies["Company Name"] == stock, "Ticker"].values[0]
            d = load_data(tkr)
            if d.empty:
                continue
            r = 100 * d["Close"].pct_change().dropna()
            try:
                m = arch_model(r, p=1, q=1)
                res = m.fit(disp="off")
                f = res.forecast(horizon=5)
                v = float(np.sqrt(f.variance.iloc[-1, 0]))
                results.append([stock, v])
            except Exception:
                pass
        if results:
            multi_df = pd.DataFrame(results, columns=["Stock", "Forecast Volatility"])
            st.line_chart(multi_df.set_index("Stock"))

# ================= SECTOR ANALYSIS ================= #

with tab3:
    st.subheader("Sector Volatility Comparison")

    sector_data = companies[companies["Industry"] == industry]

    sector_names = []
    sector_vol = []

    for _, r in sector_data.iterrows():
        try:
            d = load_data(r["Ticker"])
            ret = 100 * d["Close"].pct_change().dropna()
            m = arch_model(ret, p=1, q=1)
            res = m.fit(disp="off")
            f = res.forecast(horizon=5)
            v = float(np.sqrt(f.variance.iloc[-1, 0]))
            sector_vol.append(v)
            sector_names.append(r["Company Name"])
        except Exception:
            pass

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=sector_names,
        y=sector_vol,
        mode="lines+markers"
    ))
    fig3.update_layout(
        xaxis_title="Companies",
        yaxis_title="Volatility"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ================= SECTOR HEATMAP ================= #

with tab4:
    st.subheader("Sector Volatility Heatmap")

    heat_data = []

    for _, r in companies.iterrows():
        try:
            d = load_data(r["Ticker"])
            ret = 100 * d["Close"].pct_change().dropna()
            m = arch_model(ret, p=1, q=1)
            res = m.fit(disp="off")
            f = res.forecast(horizon=5)
            v = float(np.sqrt(f.variance.iloc[-1, 0]))
            heat_data.append([r["Industry"], r["Company Name"], v])
        except Exception:
            pass

    heat_df = pd.DataFrame(heat_data, columns=["Sector", "Company", "Volatility"])
    pivot = heat_df.pivot(index="Sector", columns="Company", values="Volatility")
    st.dataframe(pivot)

# ================= PORTFOLIO ANALYTICS ================= #

with tab5:
    st.subheader("Portfolio Volatility Analytics")

    portfolio = st.multiselect("Select Portfolio Stocks", companies["Company Name"])

    if portfolio:
        returns_df = pd.DataFrame()

        for stock in portfolio:
            tkr = companies.loc[companies["Company Name"] == stock, "Ticker"].values[0]
            d = load_data(tkr)
            returns_df[stock] = d["Close"].pct_change()

        returns_df = returns_df.dropna()

        weights = np.ones(len(portfolio)) / len(portfolio)

        cov_matrix = returns_df.cov()

        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        st.metric("Portfolio Volatility", round(port_vol, 4))

        st.line_chart(returns_df)

# ================= FACTOR EXPOSURE ================= #

with tab6:
    st.subheader("Market Beta (Factor Exposure)")

    market = "^NSEI"

    market_data = load_data(market)

    if not market_data.empty:
        stock_returns = data["Close"].pct_change().dropna()
        market_returns = market_data["Close"].pct_change().dropna()

        combined = pd.concat([stock_returns, market_returns], axis=1).dropna()
        combined.columns = ["Stock", "Market"]

        beta = np.cov(combined["Stock"], combined["Market"])[0][1] / np.var(combined["Market"])

        st.metric("Market Beta", round(beta, 3))

# ================= FINANCIAL RATIOS ================= #

with tab7:
    st.subheader("Calculated Financial Ratios")

    ticker_obj = yf.Ticker(ticker)

    try:
        financials = ticker_obj.financials
        balance = ticker_obj.balance_sheet

        net_income = financials.loc["Net Income"].iloc[0]
        revenue = financials.loc["Total Revenue"].iloc[0]
        equity = balance.loc["Total Stockholder Equity"].iloc[0]
        debt = balance.loc["Total Debt"].iloc[0]

        roe = net_income / equity if equity else np.nan
        profit_margin = net_income / revenue if revenue else np.nan
        debt_to_equity = debt / equity if equity else np.nan

        ratio_df = pd.DataFrame({
            "Ratio": ["Return on Equity", "Profit Margin", "Debt to Equity"],
            "Value": [roe, profit_margin, debt_to_equity]
        })

        st.table(ratio_df)

    except Exception:
        st.write("Financial statement data unavailable.")

    st.subheader("Financial Ratio Guide")

    learn_df = pd.DataFrame({
        "Ratio": [
            "PE Ratio",
            "Price to Book",
            "Return on Equity",
            "Debt to Equity",
            "Profit Margin"
        ],
        "Interpretation": [
            "Measures valuation relative to earnings",
            "Compares market value to book value",
            "Shows efficiency of shareholder capital",
            "Measures financial leverage risk",
            "Shows company profitability"
        ]
    })

    st.table(learn_df)

    st.download_button(
        "Download CSV",
        data.to_csv(),
        file_name=f"{ticker}_data.csv"
    )
