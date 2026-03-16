import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model

st.set_page_config(layout="wide")

st.title("Nifty Midcap Volatility & Financial Analysis Dashboard")

# ---------------- STOCK DATABASE ---------------- #

stocks = {
"IT":{
"Persistent Systems":"PERSISTENT.NS",
"Coforge":"COFORGE.NS",
"Mphasis":"MPHASIS.NS",
"Zensar Technologies":"ZENSARTECH.NS",
"Tata Elxsi":"TATELXSI.NS"
},

"Financial Services":{
"Cholamandalam Finance":"CHOLAFIN.NS",
"AU Small Finance Bank":"AUBANK.NS",
"City Union Bank":"CUB.NS",
"LIC Housing Finance":"LICHSGFIN.NS",
"Muthoot Finance":"MUTHOOTFIN.NS"
},

"Pharma":{
"Biocon":"BIOCON.NS",
"Laurus Labs":"LAURUSLABS.NS",
"Glenmark Pharma":"GLENMARK.NS",
"Abbott India":"ABBOTINDIA.NS"
},

"Chemicals":{
"Aarti Industries":"AARTIIND.NS",
"Deepak Nitrite":"DEEPAKNTR.NS",
"PI Industries":"PIIND.NS",
"SRF":"SRF.NS"
},

"Capital Goods":{
"Voltas":"VOLTAS.NS",
"Escorts Kubota":"ESCORTS.NS",
"Tube Investments":"TIINDIA.NS"
},

"Consumer":{
"Page Industries":"PAGEIND.NS",
"Trent":"TRENT.NS",
"Indian Hotels":"INDHOTEL.NS"
},

"Auto":{
"TVS Motors":"TVSMOTOR.NS",
"Balkrishna Industries":"BALKRISIND.NS"
}
}

# flatten stock list
all_companies = {}
for sector in stocks:
    for company in stocks[sector]:
        all_companies[company] = stocks[sector][company]


# ---------------- NAVIGATION ---------------- #

st.sidebar.header("Navigation")

page = st.sidebar.selectbox(
"Choose Section",
[
"Company Analysis",
"Sector Analysis",
"Learn Concepts"
]
)

# ---------------- GARCH VOLATILITY FUNCTION ---------------- #

def garch_volatility(data):

    returns = 100 * data["Close"].pct_change().dropna()

    model = arch_model(returns, vol="Garch", p=1, q=1)

    results = model.fit(disp="off")

    forecast = results.forecast(horizon=5)

    volatility = np.sqrt(forecast.variance.iloc[-1,0])

    return volatility


def volatility_label(vol):

    if vol < 1.5:
        return "🟢 Low Volatility"

    elif vol < 3:
        return "🟡 Moderate Volatility"

    else:
        return "🔴 High Volatility"


# ---------------- COMPANY ANALYSIS ---------------- #

if page == "Company Analysis":

    st.header("Company Analysis")

    company = st.selectbox(
    "Select Nifty Midcap Company",
    list(all_companies.keys())
    )

    ticker = all_companies[company]

    data = yf.download(ticker, period="2y")

    if data.empty:

        st.error("Unable to load stock data")

    else:

        st.subheader("Stock Price Chart (Last 2 Years)")

        st.line_chart(data["Close"])

        st.subheader("GARCH Volatility Forecast")

        vol = garch_volatility(data)

        st.metric("Predicted Volatility", round(vol,3))

        st.write(volatility_label(vol))

        st.subheader("Volatility Interpretation")

        st.write("""
Low volatility indicates relatively stable price movements.

Moderate volatility indicates balanced risk-return.

High volatility suggests higher uncertainty and risk.
""")


# ---------------- SECTOR ANALYSIS ---------------- #

elif page == "Sector Analysis":

    st.header("Sector Volatility Comparison")

    sector_results = []

    for sector in stocks:

        vols = []

        for company in stocks[sector]:

            ticker = stocks[sector][company]

            data = yf.download(ticker, period="2y")

            if not data.empty:

                vol = garch_volatility(data)

                vols.append(vol)

        if vols:

            avg_vol = np.mean(vols)

            sector_results.append([sector, avg_vol])

    df = pd.DataFrame(sector_results, columns=["Sector","Average Volatility"])

    st.dataframe(df)

    st.bar_chart(df.set_index("Sector"))


# ---------------- LEARN SECTION ---------------- #

elif page == "Learn Concepts":

    st.header("Learn Financial Concepts")

    st.subheader("GARCH Model")

    st.write("""
GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
models time-varying volatility in financial markets.

Financial returns often exhibit volatility clustering —
periods of high volatility followed by high volatility.

GARCH models are widely used in risk management and
derivatives pricing.
""")

    st.subheader("Price to Earnings Ratio")

    st.write("""
Definition:
Market Price per Share divided by Earnings per Share.

Interpretation:
Higher PE may indicate growth expectations.
Lower PE may indicate undervaluation.
""")

    st.subheader("Return on Equity")

    st.write("""
Definition:
Net Income / Shareholder Equity.

Interpretation:
Higher ROE indicates efficient use of capital.
""")

    st.subheader("Debt to Equity Ratio")

    st.write("""
Definition:
Total Debt / Shareholder Equity.

Interpretation:
Higher ratio means higher financial leverage.
""")
