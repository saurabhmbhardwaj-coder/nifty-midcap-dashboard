import pandas as pd

def create_heatmap(data):

    df = pd.DataFrame(data,
    columns=["Sector","Stock","Volatility"])

    return df.pivot(
        index="Sector",
        columns="Stock",
        values="Volatility"
    )
