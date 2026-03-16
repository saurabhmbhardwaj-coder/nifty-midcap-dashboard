from arch import arch_model
import numpy as np

def garch_volatility(price):

    returns = 100 * price.pct_change().dropna()

    model = arch_model(
        returns,
        vol="Garch",
        p=1,
        q=1
    )

    result = model.fit(disp="off")

    forecast = result.forecast(horizon=5)

    vol = np.sqrt(forecast.variance.iloc[-1,0])

    return float(vol)
