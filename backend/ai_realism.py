import numpy as np
import math

def compute_root_features(segments):
    lengths, angles, thicknesses, xs = [], [], [], []

    for r in segments:
        lengths.append(math.dist((r["x1"], r["y1"]), (r["x2"], r["y2"])))
        angles.append(abs(math.atan2(r["y2"] - r["y1"], r["x2"] - r["x1"])))
        thicknesses.append(r["thickness"])
        xs.append(abs(r["x2"]))

    return {
        "total_length": sum(lengths),
        "branch_count": len(segments),
        "avg_curvature": np.mean(angles),
        "thickness_variance": np.std(thicknesses),
        "horizontal_spread": max(xs),
    }


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
