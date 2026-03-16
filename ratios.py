import yfinance as yf

def get_ratios(ticker):

    info = yf.Ticker(ticker).info

    ratios = {

        "PE Ratio":info.get("trailingPE"),

        "Price to Book":info.get("priceToBook"),

        "ROE":info.get("returnOnEquity"),

        "Debt to Equity":info.get("debtToEquity"),

        "Profit Margin":info.get("profitMargins")

    }

    return ratios
