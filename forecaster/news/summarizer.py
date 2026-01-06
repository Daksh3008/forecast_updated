import numpy as np

def top_headlines(news, scores, top_k=10):
    """
    Rank headlines by absolute sentiment strength.
    """

    if not news:
        return []

    ranked = sorted(
        zip(news, scores),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top = ranked[:top_k]

    lines = []
    for i, (item, score) in enumerate(top, 1):
        title = item.get("title", "").strip()
        source = item.get("source", "Unknown")
        lines.append(f"{i}. {title} - {source}")

    return lines
