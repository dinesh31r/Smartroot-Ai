import os
import time
import streamlit as st
import requests
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# -------------------------------------------------
# GROQ API CLIENT (FREE TIER - Works Worldwide)
# -------------------------------------------------
# Free tier: 30 requests/minute, 14,400 requests/day
# Get your free API key at: https://console.groq.com/keys
# Models: llama-3.3-70b-versatile, mixtral-8x7b-32768

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def get_groq_api_key():
    """Get Groq API key from Streamlit secrets or environment variable"""
    try:
        # Try Streamlit secrets first
        api_key = st.secrets.get("groq", {}).get("api_key")
    except:
        api_key = None
    
    # Fall back to environment variable
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
    
    if api_key and api_key != "YOUR_GROQ_API_KEY":
        return api_key
    return None


def call_groq_api(prompt, max_tokens=500):
    """Call Groq API with the given prompt"""
    api_key = get_groq_api_key()
    if not api_key:
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        
        # Extract text from Groq response (OpenAI-compatible format)
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            content = message.get("content", "")
            if content:
                return content.strip()
        return None
    except Exception as e:
        print(f"Groq API error: {e}")
        return None


def llm_biological_analysis(metrics, soil_type):
    """
    LLM is used ONLY for biological reasoning and explanation.
    Uses Groq API for free analysis.
    """
    
    api_key = get_groq_api_key()
    if not api_key:
        return generate_fallback_analysis(metrics, soil_type)

    prompt = f"""
    You are a Vetiver grass (Chrysopogon zizanioides) root system expert.
    This application is specifically designed for Vetiver plant analysis.

    Soil type: {soil_type}
    Root growth metrics:
    {metrics}

    Explain in simple scientific terms:
    1. Whether this Vetiver root structure is biologically realistic
    2. Why the branching and depth occurred in this soil (Vetiver is known for deep, fibrous roots)
    3. One realistic improvement suggestion for Vetiver root health
    
    Keep your response concise (under 200 words).
    """

    try:
        result = call_groq_api(prompt, max_tokens=500)
        if result:
            return result
        return generate_fallback_analysis(metrics, soil_type)
    except Exception as e:
        print(f"Groq API error: {e}")
        return generate_fallback_analysis(metrics, soil_type)


def llm_health_check():
    """Check if Groq API is reachable"""
    api_key = get_groq_api_key()
    if not api_key:
        return {"ok": False, "error": "GROQ_API_KEY not set"}

    try:
        result = call_groq_api("Say 'OK' in one word.", max_tokens=10)
        if result:
            return {"ok": True, "model": GROQ_MODEL}
        return {"ok": False, "error": "Empty response"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def llm_explain_plant_analysis(species, health_score, moisture, nutrient, stress_label):
    """
    Generate a detailed explanation of plant analysis using Groq.
    Returns plant type description and health insights.
    """
    api_key = get_groq_api_key()
    if not api_key:
        return generate_fallback_plant_explanation(species, health_score, moisture, nutrient, stress_label)
    
    prompt = f"""
    You are a Vetiver grass (Chrysopogon zizanioides) expert. This application only analyzes Vetiver plants.

    Detected Species: {species}
    Health Score: {health_score}/100
    Moisture Level: {moisture}%
    Nutrient Level: {nutrient}%
    Stress Status: {stress_label}

    If the species is NOT Vetiver, note that the analysis may be inaccurate.
    
    Provide a brief analysis (under 150 words) covering:
    1. Vetiver grass characteristics (tall perennial grass, dense clumping, deep roots up to 3-4 meters)
    2. Current health assessment based on the metrics
    3. One actionable recommendation specific to Vetiver care

    Format your response as clear paragraphs, not bullet points.
    """

    try:
        result = call_groq_api(prompt, max_tokens=300)
        if result:
            return result
        return generate_fallback_plant_explanation(species, health_score, moisture, nutrient, stress_label)
    except Exception as e:
        print(f"Groq plant explanation error: {e}")
        return generate_fallback_plant_explanation(species, health_score, moisture, nutrient, stress_label)


def llm_explain_root_analysis(species, health_index, water_efficiency, nutrient_efficiency, root_type):
    """
    Generate a detailed explanation of root analysis using Groq.
    Returns root type description and health insights.
    """
    api_key = get_groq_api_key()
    if not api_key:
        return generate_fallback_root_explanation(species, health_index, water_efficiency, nutrient_efficiency, root_type)
    
    prompt = f"""
    You are a Vetiver grass root system expert. This application only analyzes Vetiver roots.

    Root Species: {species}
    Root Type: {root_type}
    Root Health Index: {health_index}/100
    Water Efficiency: {water_efficiency}%
    Nutrient Efficiency: {nutrient_efficiency}%

    If the root is NOT Vetiver, note that the analysis may be inaccurate.

    Provide a brief analysis (under 150 words) covering:
    1. Vetiver root characteristics (deep fibrous roots, can reach 3-4m depth, excellent for erosion control)
    2. Current root health assessment based on the metrics
    3. One actionable recommendation for Vetiver root improvement

    Format your response as clear paragraphs, not bullet points.
    """

    try:
        result = call_groq_api(prompt, max_tokens=300)
        if result:
            return result
        return generate_fallback_root_explanation(species, health_index, water_efficiency, nutrient_efficiency, root_type)
    except Exception as e:
        print(f"Groq root explanation error: {e}")
        return generate_fallback_root_explanation(species, health_index, water_efficiency, nutrient_efficiency, root_type)


# -------------------------------------------------
# FALLBACK ANALYSIS FUNCTIONS
# -------------------------------------------------

def generate_fallback_analysis(metrics, soil_type):
    """Fallback analysis when LLM is unavailable."""
    max_depth = metrics.get("max_depth_cm", 0)
    total_length = metrics.get("total_length_cm", 0)
    spread = metrics.get("horizontal_spread_cm", 0)
    hair_count = metrics.get("root_hair_count", 0)
    realism = metrics.get("realism_score", 0)
    
    analysis = f"""
**Root System Analysis** ({soil_type} Soil)

**Structure Assessment:**
• Maximum root depth: {max_depth} cm - {'Deep penetration' if max_depth > 20 else 'Shallow rooting'}
• Total root length: {total_length} cm - {'Extensive branching' if total_length > 80 else 'Limited branching'}
• Horizontal spread: {spread} cm - {'Wide lateral spread' if spread > 12 else 'Narrow spread'}
• Root hair density: {hair_count} units

**Biological Realism:** {realism}% realistic

**Soil-Specific Adaptation:**
{get_soil_specific_analysis(soil_type, max_depth)}

**Growth Pattern:**
This root system demonstrates {'strong' if realism > 70 else 'moderate' if realism > 50 else 'limited'} biological realism in {soil_type.lower()} soil conditions.
"""
    return analysis


def generate_fallback_plant_explanation(species, health_score, moisture, nutrient, stress_label):
    """Fallback plant explanation when LLM is unavailable - Vetiver focused."""
    health_status = "excellent" if health_score >= 80 else "good" if health_score >= 60 else "concerning" if health_score >= 40 else "poor"
    
    is_vetiver = "vetiver" in species.lower()
    species_note = "" if is_vetiver else f"\n\n⚠️ Note: This application is optimized for Vetiver Grass. The detected species ({species}) may result in less accurate analysis."
    
    return f"""
**Vetiver Grass (Chrysopogon zizanioides)** is a perennial bunchgrass known for its deep, dense root system that can reach 3-4 meters depth. It's widely used for soil stabilization, erosion control, and phytoremediation.

Your plant shows {health_status} overall health with a score of {health_score}/100. Current moisture level is at {moisture}% and nutrient absorption is at {nutrient}%. The stress analysis indicates: {stress_label}.

{"Vetiver is drought-tolerant - continue your current care routine." if health_score >= 60 else "Consider checking soil drainage and ensuring adequate sunlight for optimal Vetiver growth."}{species_note}
"""


def generate_fallback_root_explanation(species, health_index, water_efficiency, nutrient_efficiency, root_type):
    """Fallback root explanation when LLM is unavailable - Vetiver focused."""
    health_status = "excellent" if health_index >= 80 else "good" if health_index >= 60 else "fair" if health_index >= 40 else "poor"
    
    is_vetiver = "vetiver" in species.lower()
    species_note = "" if is_vetiver else f"\n\n⚠️ Note: This application is optimized for Vetiver roots. Non-Vetiver roots ({species}) may show less accurate metrics."
    
    return f"""
**Vetiver Root System** is renowned for its exceptional depth (up to 3-4 meters) and dense, fibrous structure. The {root_type} architecture provides excellent soil binding, erosion control, and water filtration capabilities.

Root health index is {health_status} at {health_index}/100. Water uptake efficiency is at {water_efficiency}% and nutrient absorption efficiency is at {nutrient_efficiency}%. These metrics indicate {'optimal' if health_index >= 70 else 'adequate' if health_index >= 50 else 'suboptimal'} root function.

{"The Vetiver root system appears healthy. Maintain current soil conditions for continued deep root development." if health_index >= 60 else "Consider improving soil aeration and reducing compaction to enhance Vetiver root penetration."}{species_note}
"""


def get_soil_specific_analysis(soil_type, depth):
    """Generate soil-type specific analysis."""
    if soil_type == "Sandy":
        return "• Sandy soil: Supports deep penetration with minimal resistance\n• Good drainage encourages downward growth\n• Recommend lateral expansion for moisture retention"
    elif soil_type == "Clay":
        return "• Clay soil: Dense structure requires stronger root pressure\n• Limited deep penetration expected\n• Branching at shallow depths is adaptive"
    else:  # Loamy
        return "• Loamy soil: Ideal balance supports diverse root architecture\n• Supports both deep and lateral growth\n• Excellent conditions for root development"
