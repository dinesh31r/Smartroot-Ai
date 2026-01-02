import random
import math

def simulate_root(moisture, nutrient, soil_type, steps=380):
    roots = []

    # Initial root (top → down)
    queue = [(0.0, 0.0, math.pi / 2, 4.2, 0, steps)]

    soil_params = {
        "Sandy": {"curve": 0.25, "branch": 0.35},
        "Clay":  {"curve": 0.45, "branch": 0.55},
        "Loamy": {"curve": 0.35, "branch": 0.45}
    }

    cfg = soil_params[soil_type]
    max_depth = steps

    while queue:
        x, y, angle, thickness, depth, remaining = queue.pop(0)
        if remaining <= 0 or thickness < 0.3:
            continue

        step = random.uniform(0.6, 1.2)

        # 🌱 Smooth curvature (NO sharp kinks)
        curve = random.uniform(-cfg["curve"], cfg["curve"]) * 0.6
        angle += curve

        nx = x + step * math.cos(angle)
        ny = y + step * math.sin(angle)

        roots.append({
            "x1": x, "y1": y,
            "x2": nx, "y2": ny,
            "thickness": thickness
        })

        # Continue main root
        queue.append((
            nx, ny,
            angle,
            thickness * 0.985,
            depth + step,
            remaining - 1
        ))

        # 🌿 Depth-aware branching (realistic)
        depth_factor = min(depth / max_depth, 1.0)
        if random.random() < cfg["branch"] * depth_factor:
            queue.append((
                nx, ny,
                angle + random.uniform(-0.6, 0.6),   # smooth branch angle
                thickness * 0.78,                    # slower taper
                depth + step,
                remaining * 0.6
            ))

    return roots
