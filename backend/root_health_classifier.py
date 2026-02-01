def _clamp(value, low=0.0, high=100.0):
    return max(low, min(high, value))


def _risk_level(score):
    if score < 0.25:
        return "Low"
    if score < 0.55:
        return "Medium"
    return "High"


def classify_root_health(traits):
    symmetry = traits.get("symmetry_index", 0.5)
    thickness = traits.get("thickness_mean", 2.5)
    biomass_level = traits.get("biomass_level", "Medium")
    branch_density = traits.get("branch_density", 0.003)

    brown_ratio = traits.get("brown_ratio", 0.0)
    dark_ratio = traits.get("dark_ratio", 0.0)
    fragmentation = traits.get("fragmentation", 0.0)

    thickness_norm = min(thickness / 6.0, 1.0)
    biomass_norm = {"Low": 0.3, "Medium": 0.6, "High": 0.9}.get(biomass_level, 0.6)
    branch_norm = min(branch_density / 0.008, 1.0)

    # Vetiver roots are naturally brownish/yellow. 
    # Reduced penalty for brown to avoid false positives on healthy roots.
    # Dark ratio (necrosis) remains high penalty.
    penalty_sym = (1.0 - symmetry) * 18.0
    penalty_rot = brown_ratio * 15.0 + dark_ratio * 40.0
    penalty_frag = fragmentation * 15.0
    penalty_thin = (1.0 - thickness_norm) * 10.0
    
    # NEW: Penalty for sparse/low biomass roots
    # If biomass is low, it indicates poor growth even if not rotten.
    biomass_score = {"Low": 0.2, "Medium": 0.8, "High": 1.0}.get(biomass_level, 0.5)
    penalty_biomass = (1.0 - biomass_score) * 20.0

    health_score = 100.0 - (penalty_sym + penalty_rot + penalty_frag + penalty_thin + penalty_biomass)
    health_score = int(_clamp(health_score, 0.0, 100.0))

    if health_score >= 75:
        health_status = "Healthy"
    elif health_score >= 50:
        health_status = "Moderate Stress"
    else:
        health_status = "Severe Stress"

    root_rot_score = (brown_ratio * 0.7 + dark_ratio * 0.3)
    fungal_score = brown_ratio * 0.6 + fragmentation * 0.4
    damage_score = fragmentation * 0.7 + max(0.0, 0.5 - symmetry) * 0.6


    water_eff = 40 + (branch_norm * 15) + (thickness_norm * 25) + (biomass_norm * 15) + (health_score * 0.05)
    nutrient_eff = 35 + (branch_norm * 35) + (symmetry * 15) + (health_score * 0.05)

    water_eff = int(_clamp(water_eff, 0.0, 100.0))
    nutrient_eff = int(_clamp(nutrient_eff, 0.0, 100.0))

    return {
        "health_status": health_status,
        "root_health_index": health_score,
        "water_efficiency": water_eff,
        "nutrient_efficiency": nutrient_eff,
        "disease_risk": {
            "root_rot": _risk_level(root_rot_score),
            "fungal": _risk_level(fungal_score),
            "damage": _risk_level(damage_score)
        }
    }
