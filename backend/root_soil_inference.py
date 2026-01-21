def infer_soil_traits(traits):
    aspect_ratio = traits.get("aspect_ratio", 1.0)
    branch_density = traits.get("branch_density", 0.003)
    length_density = traits.get("length_density", 0.2)
    biomass_level = traits.get("biomass_level", "Medium")

    if aspect_ratio > 1.4 and branch_density < 0.003:
        soil_type = "Sandy"
    elif aspect_ratio < 0.9 and branch_density >= 0.004:
        soil_type = "Clay"
    else:
        soil_type = "Loamy"

    compaction_score = 0.0
    if biomass_level == "Low":
        compaction_score += 0.4
    if length_density < 0.2:
        compaction_score += 0.35
    if branch_density < 0.003:
        compaction_score += 0.25

    if compaction_score < 0.35:
        soil_compaction = "Low"
    elif compaction_score < 0.7:
        soil_compaction = "Medium"
    else:
        soil_compaction = "High"

    return {
        "soil_type": soil_type,
        "soil_compaction": soil_compaction
    }
