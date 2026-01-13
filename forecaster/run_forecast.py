# forecaster/run_forecast.py
'''
predict price of crude after 2 months
after 10 days
in 3 weeks
Q1 2026
next quarter
January 2026
28 Jan 2026
01/28/2026
next month
current week
'''

import sys

from forecaster.forecast_engine import run_forecast
from forecaster.Predict import calculate_date


def _ask_forecast_intent():
    """
    Ask user for natural language forecast intent.
    """
    print("🔮 Brent Crude Forecast Agent")
    print("Examples:")
    print("  - predict price of crude after 2 months")
    print("  - after 10 days")
    print("  - Q1 2026")
    print("  - next quarter")
    print("")

    return input("Enter your forecast request: ").strip()


def main():
    # --------------------------------------------------
    # 1. Get natural language intent
    # --------------------------------------------------
    user_input = _ask_forecast_intent()

    resolved_date = calculate_date(user_input)

    if isinstance(resolved_date, str):
        print(f"❌ Date agent error: {resolved_date}")
        sys.exit(1)

    print(f"\n📅 Using target date: {resolved_date}")

    # --------------------------------------------------
    # 2. Run forecast (NO DATE MODIFICATION)
    # --------------------------------------------------
    result = run_forecast(
        asset="brent",
        target_date=resolved_date
    )

    # --------------------------------------------------
    # 3. Print report
    # --------------------------------------------------
    print("\n" + "=" * 80)
    print("🔮 BRENT CRUDE FORECAST")
    print("=" * 80)
    print(result["text_report"])
    print("=" * 80)

    print(f"\nSaved outputs to:")
    print(f" - {result['txt_path']}")
    print(f" - {result['json_path']}")
    print(f" - {result['chart_path']}")

if __name__ == "__main__":
    main()


