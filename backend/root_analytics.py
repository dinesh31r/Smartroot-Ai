"""
Safe root analytics (no divide-by-zero)
"""

import math

def analyze_roots(segments):
    if not segments:
        return {
            "Max Root Depth (cm)": 0,
            "Total Root Length (cm)": 0,
            "Average Root Thickness": 0,
            "Branching Density": 0,
            "Root Health Index": 0
        }

    total_length = 0
    max_depth = 0
    thickness_sum = 0

    for s in segments:
        total_length += math.dist((s["x1"], s["y1"]), (s["x2"], s["y2"]))
        max_depth = max(max_depth, abs(s["y2"]))
        thickness_sum += s["thickness"]

    avg_thickness = thickness_sum / len(segments)
    safe_depth = max(max_depth, 0.1)

    branching_density = len(segments) / safe_depth

    health_index = round(
        (avg_thickness * 0.35) +
        (branching_density * 0.25) +
        (total_length * 0.40),
        2
    )

    return {
        "Max Root Depth (cm)": round(safe_depth * 10, 2),
        "Total Root Length (cm)": round(total_length * 10, 2),
        "Average Root Thickness": round(avg_thickness, 2),
        "Branching Density": round(branching_density, 3),
        "Root Health Index": health_index
    }
