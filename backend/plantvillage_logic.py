"""
PlantVillage Logic Integration
SmartRoot-AI
Provides color-based analysis for Moisture and Nutrient stress.
"""

import cv2
import numpy as np

# NEW (SAFE ADDITION)
def analyze_plant_health_logic(image):
    """
    Analyzes plant health using color-based logic (PlantVillage principles).
    Detects Chlorosis (Yellowing) and Necrosis (Browning).
    """
    if image is None:
        return {}

    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define ranges for "Healthy Green", "Yellow (Nutrient Stress)", and "Brown (Moisture Stress)"
    # HSV: Hue (0-180), Saturation (0-255), Value (0-255)
    
    # Healthy Green
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    
    # Yellowing (Chlorosis) -> Typical of Nutrient Deficiency
    yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([34, 255, 255]))
    
    # Brown/Dry (Necrosis) -> Typical of Moisture Stress
    brown_mask = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([20, 255, 200]))
    
    total_pixels = image.shape[0] * image.shape[1]
    green_area = np.sum(green_mask > 0)
    yellow_area = np.sum(yellow_mask > 0)
    brown_area = np.sum(brown_mask > 0)
    
    # Calculate ratios relative to green area (plant body)
    plant_area = max(green_area + yellow_area + brown_area, 1)
    
    yellow_ratio = (yellow_area / plant_area) * 100
    brown_ratio = (brown_area / plant_area) * 100
    
    # Nutrient Score: Starts at 100, drops as yellowing increases
    nutrient_score = max(0, min(100, int(85 - (yellow_ratio * 2))))
    
    # Moisture Score: Starts at 100, drops as browning increases
    moisture_score = max(0, min(100, int(90 - (brown_ratio * 3))))
    
    # Determine Status
    status = "Healthy"
    if yellow_ratio > 15:
        status = "Low Nutrient (Logic)"
    if brown_ratio > 10:
        status = "Low Moisture (Logic)"
    if yellow_ratio > 30 and brown_ratio > 20:
        status = "Severely Stressed (Logic)"

    return {
        "logic_moisture": moisture_score,
        "logic_nutrient": nutrient_score,
        "logic_status": status,
        "yellow_index": round(yellow_ratio, 2),
        "brown_index": round(brown_ratio, 2),
        "logic_active": True
    }
