import os
import feedparser
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict

# ------------------------------------------------------------
# RSS SOURCES (HIGH SIGNAL FOR OIL / MACRO / GEOPOLITICS)
# ------------------------------------------------------------

RSS_FEEDS = [
    # --- Google News (broad aggregator) ---
    "https://news.google.com/rss/search?q=brent+crude+oil",
    "https://news.google.com/rss/search?q=OPEC+oil",
    "https://news.google.com/rss/search?q=oil+prices+global",
    "https://news.google.com/rss/search?q=oil+geopolitics",
    "https://news.google.com/rss/search?q=global+oil+demand",
    "https://news.google.com/rss/search?q=oil+supply+cuts",

    # --- Energy / Commodities specialists ---
    "https://www.reuters.com/markets/commodities/rss",
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",   # energy
    "https://oilprice.com/rss/main",
    "https://www.fxempire.com/feed",
    "https://www.investing.com/rss/news_95.rss",              # commodities

    # --- Macro / geopolitics ---
    "https://www.ft.com/rss/commodities",
    "https://www.ft.com/rss/world",
    "https://www.bloomberg.com/energy/rss",                    # may partially fail, safe
    "https://www.economist.com/finance-and-economics/rss.xml",
]

DATA_DIR = "data"
NEWS_CSV_PATH = os.path.join(DATA_DIR, "news.csv")


# ------------------------------------------------------------
# CORE FETCHER
# ------------------------------------------------------------

def fetch_news(days: int = 60) -> List[Dict]:
    """
    Fetch maximum available oil/macro/geopolitical news via RSS.

    Returns:
        List[dict] with keys:
        - title
        - source
        - published (datetime)
        - date (date)
    """

    cutoff = datetime.utcnow() - timedelta(days=days)
    records = []

    for url in RSS_FEEDS:
        feed = feedparser.parse(url)

        for entry in feed.entries:
            # -------- published date --------
            try:
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    published = datetime(*entry.updated_parsed[:6])
                else:
                    continue
            except Exception:
                continue

            if published < cutoff:
                continue

            title = entry.title.strip() if hasattr(entry, "title") else ""
            if not title:
                continue

            source = (
                entry.source.title
                if hasattr(entry, "source") and hasattr(entry.source, "title")
                else feed.feed.get("title", "Unknown")
            )

            records.append({
                "title": title,
                "source": source,
                "published": published,
                "date": published.date(),
            })

    # --------------------------------------------------------
    # DEDUPLICATION (TITLE-BASED, CASE-INSENSITIVE)
    # --------------------------------------------------------

    seen = set()
    unique = []

    for r in sorted(records, key=lambda x: x["published"], reverse=True):
        key = r["title"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # --------------------------------------------------------
    # PERSIST ALL NEWS (AUDITABLE)
    # --------------------------------------------------------

    os.makedirs(DATA_DIR, exist_ok=True)

    if unique:
        df = pd.DataFrame(unique)
        df.sort_values("published", ascending=False, inplace=True)
        df.to_csv(NEWS_CSV_PATH, index=False)

    return unique
