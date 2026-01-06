def build_macro_context(m):
    lines = []

    if m.get("usdinr"):
        if m["usdinr"] > 85:
            lines.append(f"• USD/INR at {m['usdinr']:.2f} indicates INR weakness, increasing import cost pressure.")
        else:
            lines.append(f"• USD/INR at {m['usdinr']:.2f} remains stable.")

    if m.get("vix"):
        if m["vix"] < 15:
            lines.append(f"• VIX at {m['vix']:.1f} suggests subdued risk aversion.")
        else:
            lines.append(f"• Elevated VIX at {m['vix']:.1f} indicates rising market uncertainty.")

    if m.get("brent") and m.get("wti"):
        spread = m["brent"] - m["wti"]
        if abs(spread) < 3:
            lines.append("• Brent–WTI spread remains narrow, indicating balanced global supply.")
        else:
            lines.append("• Wider Brent–WTI spread reflects regional supply–demand imbalances.")

    if m.get("us10y"):
        if m["us10y"] > 4:
            lines.append(f"• US 10Y yields near {m['us10y']:.2f}% suggest tight financial conditions.")
        else:
            lines.append(f"• US 10Y yields around {m['us10y']:.2f}% remain accommodative.")

    return lines
