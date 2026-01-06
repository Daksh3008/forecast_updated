import numpy as np


def market_regime(macro, price_series):
    lines = []

    vix = macro.get("vix")
    vol = price_series.pct_change().rolling(20).std().iloc[-1]
    slope = np.polyfit(range(20), price_series.tail(20), 1)[0]

    if vix and vix < 15 and vol < 0.02:
        regime = "stable"
    elif vix and vix > 25:
        regime = "risk-off"
    elif vol > 0.03:
        regime = "volatile"
    else:
        regime = "transitional"

    if regime == "stable" and slope < 0:
        summary = (
            "Markets remain in a stable regime, but Brent prices continue "
            "to drift lower as supply pressures ease and volatility remains contained."
        )
    elif regime == "risk-off":
        summary = (
            "Markets are in a risk-off regime, with elevated uncertainty "
            "driving cautious positioning in crude oil."
        )
    elif regime == "volatile":
        summary = (
            "Crude markets are in a volatile regime, with sharp price swings "
            "reflecting sensitivity to macro and geopolitical developments."
        )
    else:
        summary = (
            "Markets appear to be in a transitional regime, with mixed signals "
            "from price trends and volatility."
        )

    return regime, summary
