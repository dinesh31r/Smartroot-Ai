"""
Vetiver Root Realism Evaluator
SmartRoot-AI

Evaluates biological realism of simulated Vetiver root systems.
"""

def evaluate_root_realism(segments):
    """
    Evaluate the biological realism of a Vetiver root system.
    
    Args:
        segments: List of root segments from simulate_root()
        
    Returns:
        tuple: (score 0-100, list of feedback strings)
    """
    if not segments or not isinstance(segments, list):
        return 50, ["Invalid input data"]
    
    # Validate segment structure
    if not all(isinstance(r, dict) and "thickness" in r for r in segments):
        return 50, ["Missing thickness data in segments"]
    
    score = 100
    feedback = []
    
    # Filter out root hairs for main analysis
    main_roots = [r for r in segments if r.get("segment_type") != "root_hair"]
    root_hairs = [r for r in segments if r.get("segment_type") == "root_hair"]
    
    if not main_roots:
        return 40, ["No main root segments found"]
    
    total = len(main_roots)
    
    # === THICKNESS DISTRIBUTION ===
    thin = sum(1 for r in main_roots if r["thickness"] < 0.7)
    medium = sum(1 for r in main_roots if 0.7 <= r["thickness"] < 2.0)
    thick = sum(1 for r in main_roots if r["thickness"] >= 2.0)
    
    thin_ratio = thin / total
    thick_ratio = thick / total
    
    # Vetiver should have ~85-95% fine roots
    if thin_ratio < 0.30:
        score -= 20
        feedback.append("Too few fine roots (Vetiver has dense fibrous system)")
    elif thin_ratio < 0.60:
        score -= 10
        feedback.append("Fine root density below typical Vetiver levels")
    elif thin_ratio >= 0.85:
        feedback.append("✓ Excellent fine root density (typical Vetiver)")
    
    # Primary roots should be <5% of total
    if thick_ratio > 0.15:
        score -= 15
        feedback.append("Too many thick primary roots")
    elif thick_ratio > 0.08:
        score -= 5
        feedback.append("Slightly high primary root proportion")
    else:
        feedback.append("✓ Good primary/secondary root balance")
    
    # === DEPTH ANALYSIS ===
    max_depth = max(r.get("depth_cm", r["y2"]) for r in main_roots)
    
    if max_depth < 50:
        score -= 15
        feedback.append(f"Shallow rooting ({max_depth:.0f}cm) - Vetiver typically reaches 1-3m")
    elif max_depth < 100:
        score -= 5
        feedback.append(f"Moderate depth ({max_depth:.0f}cm) - healthy Vetiver goes deeper")
    elif max_depth >= 150:
        feedback.append(f"✓ Good depth penetration ({max_depth:.0f}cm)")
    
    # === ROOT HAIR PRESENCE (Vetiver characteristic) ===
    if len(root_hairs) == 0:
        score -= 5
        feedback.append("No root hairs detected")
    elif len(root_hairs) < total * 0.05:
        feedback.append("Low root hair density")
    else:
        feedback.append(f"✓ Good root hair development ({len(root_hairs)} hairs)")
    
    # === BRANCHING PATTERN ===
    generations = [r.get("generation", 0) for r in main_roots if r.get("generation", 0) < 99]
    if generations:
        max_gen = max(generations)
        if max_gen < 2:
            score -= 10
            feedback.append("Limited branching complexity")
        elif max_gen >= 4:
            feedback.append("✓ Complex branching architecture")
    
    # === SEGMENT TYPE DISTRIBUTION ===
    taproots = sum(1 for r in main_roots if r.get("segment_type") == "taproot")
    primary = sum(1 for r in main_roots if r.get("segment_type") == "primary")
    secondary = sum(1 for r in main_roots if r.get("segment_type") == "secondary")
    tertiary = sum(1 for r in main_roots if r.get("segment_type") == "tertiary")
    
    if tertiary > secondary > primary:
        feedback.append("✓ Proper hierarchical root structure")
    
    # === FINAL ASSESSMENT ===
    score = max(score, 40)  # Minimum score
    
    if score >= 90:
        feedback.insert(0, "🌱 Excellent Vetiver root realism")
    elif score >= 75:
        feedback.insert(0, "✓ Good biological accuracy")
    elif score >= 60:
        feedback.insert(0, "⚠ Moderate realism - some improvements needed")
    else:
        feedback.insert(0, "⚠ Low realism score - review parameters")
    
    return score, feedback

