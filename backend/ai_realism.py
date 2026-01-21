def evaluate_root_realism(segments):
    score = 100
    feedback = []

    thin = sum(1 for r in segments if r["thickness"] < 0.7)
    thick = sum(1 for r in segments if r["thickness"] > 2.5)

    if thin < len(segments) * 0.3:
        score -= 15
        feedback.append("Too few fine roots")

    if thick > len(segments) * 0.4:
        score -= 10
        feedback.append("Excessively thick primary roots")

    if score >= 85:
        feedback.append("Excellent biological realism")

    return max(score, 40), feedback
