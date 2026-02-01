"""
Vetiver Root Growth Simulator - SmartRoot-AI
Realistic root visualization with moisture/nutrient response
"""

import random
import math


def simulate_root(moisture, nutrient, soil_type, steps=380):
    """
    Simulate Vetiver root growth with environmental factors.
    
    Args:
        moisture: Soil moisture level (0-100%)
        nutrient: Soil nutrient level (0-100%)
        soil_type: "Sandy", "Clay", or "Loamy"
        steps: Maximum growth iterations
        
    Returns:
        List of root segments with position and thickness
    """
    roots = []
    
    # === ENVIRONMENTAL FACTORS ===
    moisture_factor = max(0.3, min(1.0, moisture / 100.0))
    nutrient_factor = max(0.3, min(1.0, nutrient / 100.0))
    
    # Low moisture = deeper roots (searching for water)
    # Low nutrients = more branching (exploring for nutrients)
    depth_multiplier = 1.0 + (1.0 - moisture_factor) * 0.5
    branch_multiplier = 1.0 + (1.0 - nutrient_factor) * 0.4
    vigor = (moisture_factor * 0.6 + nutrient_factor * 0.4)

    soil_params = {
        "Sandy": {"curve": 0.20, "branch": 0.30},
        "Clay":  {"curve": 0.40, "branch": 0.50},
        "Loamy": {"curve": 0.30, "branch": 0.40}
    }

    cfg = soil_params.get(soil_type, soil_params["Loamy"])
    adjusted_steps = int(steps * depth_multiplier * vigor)
    max_depth = adjusted_steps
    
    # === MULTIPLE PRIMARY ROOTS FROM CENTER ===
    # Vetiver has fibrous root system - multiple roots from crown
    num_primary = random.randint(5, 8)
    queue = []
    
    for i in range(num_primary):
        # Spread roots around center, but all start at (0, 0)
        spread_angle = math.pi/2 + random.uniform(-0.4, 0.4)  # Mostly downward
        thickness = random.uniform(3.5, 4.5) * vigor
        queue.append((0.0, 0.0, spread_angle, thickness, 0, adjusted_steps + random.randint(-30, 30)))

    while queue:
        x, y, angle, thickness, depth, remaining = queue.pop(0)
        if remaining <= 0 or thickness < 0.25:
            continue

        step = random.uniform(0.5, 1.0)

        # Smooth curvature
        curve = random.uniform(-cfg["curve"], cfg["curve"]) * 0.5
        
        # Gravitropism - roots tend downward
        if angle < math.pi/2:
            curve += 0.02
        elif angle > math.pi/2:
            curve -= 0.02
            
        angle += curve
        # Clamp angle to prevent upward growth
        angle = max(0.2, min(math.pi - 0.2, angle))

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
            thickness * 0.988,
            depth + step,
            remaining - 1
        ))

        # Depth-aware branching (more branching in nutrient zone)
        depth_ratio = min(depth / max(max_depth, 1), 1.0)
        
        # More branching in upper soil (nutrient rich) and when nutrients are low
        if depth_ratio < 0.4:
            branch_zone = 1.3  # Dense in top 40%
        elif depth_ratio < 0.7:
            branch_zone = 1.0
        else:
            branch_zone = 0.5  # Sparse at depth
            
        branch_prob = cfg["branch"] * branch_multiplier * branch_zone * depth_ratio * 0.8
        
        if random.random() < branch_prob:
            direction = random.choice([-1, 1])
            branch_angle = angle + direction * random.uniform(0.4, 0.8)
            queue.append((
                nx, ny,
                branch_angle,
                thickness * random.uniform(0.65, 0.80),
                depth + step,
                int(remaining * random.uniform(0.4, 0.6))
            ))

    # === ADD ROOT HAIRS ===
    root_hairs = _generate_root_hairs(roots, nutrient_factor, max_depth)
    
    return roots + root_hairs


def _generate_root_hairs(roots, nutrient_factor, max_depth):
    """Generate fine root hairs on thin roots."""
    root_hairs = []
    
    # More root hairs when nutrients are low (plant searching for nutrients)
    hair_density = 0.12 + (1.0 - nutrient_factor) * 0.15
    
    for seg in roots:
        # Only on thinner roots
        if seg["thickness"] > 1.5:
            continue
        
        # More hairs in upper soil
        depth = seg["y2"]
        if depth > max_depth * 0.6:
            continue
            
        if random.random() > hair_density:
            continue
        
        # Generate 1-3 root hairs
        for _ in range(random.randint(1, 3)):
            t = random.uniform(0.2, 0.8)
            hx = seg["x1"] + t * (seg["x2"] - seg["x1"])
            hy = seg["y1"] + t * (seg["y2"] - seg["y1"])
            
            # Perpendicular direction
            seg_angle = math.atan2(seg["y2"] - seg["y1"], seg["x2"] - seg["x1"])
            hair_angle = seg_angle + random.choice([-1, 1]) * (math.pi/2 + random.uniform(-0.2, 0.2))
            
            hair_len = random.uniform(0.3, 0.7)
            hx2 = hx + hair_len * math.cos(hair_angle)
            hy2 = hy + hair_len * math.sin(hair_angle)
            
            root_hairs.append({
                "x1": hx, "y1": hy,
                "x2": hx2, "y2": hy2,
                "thickness": random.uniform(0.08, 0.15),
                "is_hair": True
            })
    
    return root_hairs


def get_vetiver_root_stats(segments):
    """
    Calculate root system statistics.
    """
    if not segments:
        return {}
    
    max_depth_cm = max(s["y2"] for s in segments) if segments else 0
    total_length_cm = sum(
        math.sqrt((s["x2"]-s["x1"])**2 + (s["y2"]-s["y1"])**2) 
        for s in segments
    )
    max_spread_cm = max(abs(s["x2"]) for s in segments) if segments else 0
    
    return {
        "max_depth_m": round(max_depth_cm / 100, 2),
        "max_depth_cm": round(max_depth_cm, 1),
        "total_length_m": round(total_length_cm / 100, 2),
        "horizontal_spread_cm": round(max_spread_cm, 1),
        "total_segments": len(segments)
    }
