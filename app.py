import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Nifty Midcap 150 Dashboard")

# Nifty Midcap Stocks (sample large list)
stocks = {
"Persistent Systems":"PERSISTENT.NS",
"Tata Elxsi":"TATELXSI.NS",
"Voltas":"VOLTAS.NS",
"AU Small Finance Bank":"AUBANK.NS",
"Polycab":"POLYCAB.NS",
"Page Industries":"PAGEIND.NS",
"Abbott India":"ABBOTINDIA.NS",
"Tube Investments":"TIINDIA.NS",
"Aarti Industries":"AARTIIND.NS",
"Astral":"ASTRAL.NS",
"Balkrishna Industries":"BALKRISIND.NS",
"Biocon":"BIOCON.NS",
"Cholamandalam Finance":"CHOLAFIN.NS",
"City Union Bank":"CUB.NS",
"Coforge":"COFORGE.NS",
"Coromandel International":"COROMANDEL.NS",
"Deepak Nitrite":"DEEPAKNTR.NS",
"Dixon Technologies":"DIXON.NS",
"Escorts Kubota":"ESCORTS.NS",
"Glenmark Pharma":"GLENMARK.NS",
"Godrej Industries":"GODREJIND.NS",
"Grindwell Norton":"GRINDWELL.NS",
"Indian Hotels":"INDHOTEL.NS",
"Indraprastha Gas":"IGL.NS",
"JK Cement":"JKCEMENT.NS",
"L&T Finance":"LTF.NS",
"Laurus Labs":"LAURUSLABS.NS",
"LIC Housing Finance":"LICHSGFIN.NS",
"Max Healthcare":"MAXHEALTH.NS",
"Mphasis":"MPHASIS.NS",
"MRF":"MRF.NS",
"Muthoot Finance":"MUTHOOTFIN.NS",
"NHPC":"NHPC.NS",
"PI Industries":"PIIND.NS",
"Prestige Estates":"PRESTIGE.NS",
"SRF":"SRF.NS",
"Supreme Industries":"SUPREMEIND.NS",
"Tata Chemicals":"TATACHEM.NS",
"Trent":"TRENT.NS",
"TVS Motors":"TVSMOTOR.NS",
"United Breweries":"UBL.NS",
"Voltas":"VOLTAS.NS",
"Zensar Technologies":"ZENSARTECH.NS"
}

# Dropdown
company = st.selectbox(
"Select Nifty Midcap Stock",
list(stocks.keys())
)

ticker = stocks[company]

# Download data
data = yf.download(ticker, period="1y")

# Show chart
st.subheader("Stock Price Chart")

if not data.empty:
    st.line_chart(data["Close"])
else:
    st.write("No data available")

# Show raw data
st.subheader("Raw Data")

st.write(data)
