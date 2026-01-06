NEG_WORDS = {
    "decline", "fall", "drop", "weak", "slowdown", "oversupply",
    "cut", "recession", "pressure", "risk", "uncertainty"
}

POS_WORDS = {
    "rise", "gain", "strong", "recovery", "tight", "surge",
    "boost", "growth", "demand"
}


def score_headline(text):
    t = text.lower()
    score = 0

    for w in NEG_WORDS:
        if w in t:
            score -= 1
    for w in POS_WORDS:
        if w in t:
            score += 1

    if score > 0:
        return 1
    if score < 0:
        return -1
    return 0


def sentiment_summary(headlines):
    scores = [score_headline(h["title"]) for h in headlines]

    if not scores:
        return {"avg": 0.0, "pos": 0, "neg": 0, "neu": 0}

    return {
        "avg": sum(scores) / len(scores),
        "pos": scores.count(1),
        "neg": scores.count(-1),
        "neu": scores.count(0)
    }
