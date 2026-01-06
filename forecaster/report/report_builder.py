def build_report(
    model_prices,
    anchor_date,
    target_date,
    weights,
    macro_lines,
    news_lines,
    sentiment,
    bands,
    regime_text
):
    lines = []

    lines.append("=" * 80)
    lines.append("🔮 Predicting Brent Crude Price (LSTM + TCN + Ridge)")
    lines.append("=" * 80)
    lines.append("")

    lines.append("🧾 Executive Market Overview")
    lines.append("-" * 60)

    lines.append(f"Anchor date: {anchor_date}")
    lines.append(f"Target date: {target_date}")
    lines.append("")
    lines.append("Model price estimates:")
    lines.append(f"• LSTM:     ${model_prices['lstm']:.2f}")
    lines.append(f"• TCN:      ${model_prices['tcn']:.2f}")
    lines.append(f"• Ridge:    ${model_prices['ridge']:.2f}")
    lines.append(f"• Ensemble: ${model_prices['ensemble']:.2f}")
    lines.append("")

    lines.append(f"Anchor date: {anchor_date}")
    lines.append(f"Target date: {target_date}")
    lines.append("")

    lines.append("🌍 Macro Context")
    lines.append("-" * 60)
    for l in macro_lines:
        lines.append(l)
    lines.append("")

    lines.append("📈 Model Context")
    lines.append("-" * 60)
    for k, v in weights.items():
        lines.append(f"• {k.upper()}: {v:.3f}")
    lines.append("")

    lines.append("📊 Confidence Bands")
    lines.append("-" * 60)
    lines.append(
        f"68% confidence range: ${bands['68%'][0]:.2f} – ${bands['68%'][1]:.2f}"
    )
    lines.append(
        f"95% confidence range: ${bands['95%'][0]:.2f} – ${bands['95%'][1]:.2f}"
    )
    lines.append("")

    lines.append("🧭 Market Regime Summary")
    lines.append("-" * 60)
    lines.append(regime_text)
    lines.append("")


    lines.append("📰 News Summary (Last 60 days)")
    lines.append("-" * 60)
    lines.append(f"Overall sentiment score: {sentiment['avg']:.2f}")
    lines.append("Major recent headlines:")
    for i, h in enumerate(news_lines, 1):
        lines.append(f"{i}. {h}")
    lines.append("")

    lines.append("📌 Simple Summary")
    lines.append("-" * 60)
    lines.append("• Price-only models suggest current trend persistence.")
    lines.append("• Macro indicators are neutral to mildly adverse.")
    lines.append("• News flow remains cautious.")

    return "\n".join(lines)
