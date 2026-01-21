from backend.root_traits_extractor import extract_root_traits
from backend.root_health_classifier import classify_root_health
from backend.root_soil_inference import infer_soil_traits


def analyze_root_image(image_path, fast=False):
    traits = extract_root_traits(image_path, fast=fast)
    health = classify_root_health(traits)
    soil = infer_soil_traits(traits)

    return {
        "root_type": traits.get("root_type", ""),
        "branch_density": traits.get("branch_density_label", ""),
        "growth_direction": traits.get("growth_direction", ""),
        "health_status": health.get("health_status", ""),
        "age_estimate": traits.get("age_estimate", ""),
        "biomass": traits.get("biomass_level", ""),
        "water_efficiency": health.get("water_efficiency", 0),
        "nutrient_efficiency": health.get("nutrient_efficiency", 0),
        "soil_type": soil.get("soil_type", ""),
        "soil_compaction": soil.get("soil_compaction", ""),
        "symmetry_index": round(traits.get("symmetry_index", 0.0), 3),
        "root_health_index": health.get("root_health_index", 0),
        "branch_points": traits.get("branch_points", 0),
        "end_points": traits.get("end_points", 0),
        "branching_factor": round(traits.get("branching_factor", 0.0), 3),
        "root_density": round(traits.get("root_density", 0.0), 3),
        "root_length_index": round(traits.get("root_length_index", 0.0), 3),
        "avg_thickness": round(traits.get("thickness_mean", 0.0), 2),
        "thickness_variation": round(traits.get("thickness_std", 0.0), 2),
        "disease_risk": health.get("disease_risk", {"root_rot": "", "fungal": "", "damage": ""}),
        "root_area": traits.get("root_area", 0),
        "avg_root_density": round(traits.get("avg_root_density", 0.0), 3),
        "root_system_depth": traits.get("root_system_depth", 0),
        "root_system_width": traits.get("root_system_width", 0),
        "skeleton_depth": traits.get("skeleton_depth", 0),
        "skeleton_width": traits.get("skeleton_width", 0),
        "diameter_percentiles": traits.get("diameter_percentiles", {}),
        "skeleton_diameter_percentiles": traits.get("skeleton_diameter_percentiles", {}),
        "root_distribution_x": round(traits.get("root_distribution_x", 0.0), 3),
        "root_distribution_y": round(traits.get("root_distribution_y", 0.0), 3),
        "root_tip_count": traits.get("root_tip_count", 0),
        "top_angle": round(traits.get("top_angle", 0.0), 1),
        "bottom_angle": round(traits.get("bottom_angle", 0.0), 1),
        "angle_mean": round(traits.get("angle_mean", 0.0), 1),
        "angle_min": round(traits.get("angle_min", 0.0), 1),
        "angle_max": round(traits.get("angle_max", 0.0), 1),
        "adventitious_count": traits.get("adventitious_count", 0),
        "basal_count": traits.get("basal_count", 0),
        "adventitious_angle": round(traits.get("adventitious_angle", 0.0), 1),
        "basal_angle": round(traits.get("basal_angle", 0.0), 1),
        "taproot_diameter": round(traits.get("taproot_diameter", 0.0), 2),
        "hypocotyl_diameter": round(traits.get("hypocotyl_diameter", 0.0), 2),
        "cp_dia25": traits.get("cp_dia25", 0),
        "cp_dia50": traits.get("cp_dia50", 0),
        "cp_dia75": traits.get("cp_dia75", 0),
        "cp_dia90": traits.get("cp_dia90", 0),
        "nodal_length": round(traits.get("nodal_length", 0.0), 2),
        "nodal_avg_diameter": round(traits.get("nodal_avg_diameter", 0.0), 2),
        "lateral_branch_freq": round(traits.get("lateral_branch_freq", 0.0), 4),
        "lateral_avg_length": round(traits.get("lateral_avg_length", 0.0), 2),
        "lateral_angle_mean": round(traits.get("lateral_angle_mean", 0.0), 1),
        "lateral_angle_min": round(traits.get("lateral_angle_min", 0.0), 1),
        "lateral_angle_max": round(traits.get("lateral_angle_max", 0.0), 1)
    }
