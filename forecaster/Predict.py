# forecaster/Predict.py

import re
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

# ============================================================
# Helpers
# ============================================================

def _quarter_end(year, q):
    month = {1: 3, 2: 6, 3: 9, 4: 12}[q]
    return date(year, month, 1) + relativedelta(months=1, days=-1)


def _month_end(year, month):
    return date(year, month, 1) + relativedelta(months=1, days=-1)


def _print_result(d):
    print(f"📅 Calculated target date: {d}")
    return d


# ============================================================
# Main intelligent date resolver
# ============================================================

def calculate_date(user_input: str):
    """
    Robust natural-language date resolver.
    Returns datetime.date OR string error.
    """
    today = date.today()
    text = user_input.lower().strip()

    # --------------------------------------------------------
    # 1. Explicit date formats
    # --------------------------------------------------------
    for fmt in ("%d %b %Y", "%d %B %Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return _print_result(datetime.strptime(text, fmt).date())
        except ValueError:
            pass

    # --------------------------------------------------------
    # 2. Relative offsets (after / in)
    # --------------------------------------------------------
    rel_match = re.search(
        r"(after|in)\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)",
        text,
    )
    if rel_match:
        n = int(rel_match.group(2))
        unit = rel_match.group(3)

        if "day" in unit:
            return _print_result(today + timedelta(days=n))
        if "week" in unit:
            return _print_result(today + timedelta(weeks=n))
        if "month" in unit:
            return _print_result(today + relativedelta(months=n))
        if "year" in unit:
            return _print_result(today + relativedelta(years=n))

    # --------------------------------------------------------
    # 3. Quarter references
    # --------------------------------------------------------
    q_match = re.search(r"q([1-4])\s*(\d{4})", text)
    if q_match:
        q = int(q_match.group(1))
        y = int(q_match.group(2))
        return _print_result(_quarter_end(y, q))

    if "current quarter" in text:
        q = (today.month - 1) // 3 + 1
        return _print_result(_quarter_end(today.year, q))

    if "next quarter" in text:
        q = (today.month - 1) // 3 + 2
        y = today.year
        if q > 4:
            q -= 4
            y += 1
        return _print_result(_quarter_end(y, q))

    # --------------------------------------------------------
    # 4. Month references
    # --------------------------------------------------------
    month_match = re.search(
        r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})",
        text,
    )
    if month_match:
        month = datetime.strptime(month_match.group(1), "%B").month
        year = int(month_match.group(2))
        return _print_result(_month_end(year, month))

    if "next month" in text:
        d = today + relativedelta(months=1)
        return _print_result(_month_end(d.year, d.month))

    if "current month" in text:
        return _print_result(_month_end(today.year, today.month))

    # --------------------------------------------------------
    # 5. Week references
    # --------------------------------------------------------
    if "next week" in text:
        d = today + timedelta(weeks=1)
        return _print_result(d + timedelta(days=(6 - d.weekday())))

    if "current week" in text:
        return _print_result(today + timedelta(days=(6 - today.weekday())))

    # --------------------------------------------------------
    # 6. Day references
    # --------------------------------------------------------
    if "today" in text:
        return _print_result(today)

    if "tomorrow" in text:
        return _print_result(today + timedelta(days=1))

    if "yesterday" in text:
        return _print_result(today - timedelta(days=1))

    # --------------------------------------------------------
    # 7. Fallback
    # --------------------------------------------------------
    return "Unable to understand the date request"


# ============================================================
# Standalone execution (debug / demo)
# ============================================================

if __name__ == "__main__":
    print("🧠 Intelligent Date Resolver")
    print("Type anything (e.g. 'predict crude after 2 months', 'Q1 2026')")
    print("Type 'quit' to exit")

    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break

        result = calculate_date(user_input)
        if isinstance(result, str):
            print("❌", result)


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