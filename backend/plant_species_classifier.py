import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import cv2
import numpy as np
import requests
import streamlit as st


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _get_groq_api_key():
    """Get Groq API key from Streamlit secrets or environment variable"""
    try:
        api_key = st.secrets.get("groq", {}).get("api_key")
    except:
        api_key = None
    
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
    
    if api_key and api_key != "YOUR_GROQ_API_KEY":
        return api_key
    return None


def _encode_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _llm_classify_species(image_path, focus="plant"):
    """Use Groq with LLaMA 4 Scout vision model to classify species"""
    api_key = _get_groq_api_key()
    if not api_key:
        return None

    image_b64 = _encode_image_base64(image_path)
    
    prompt = (
        f"Identify the most likely {focus} species based on this image. "
        f"Focus on identifying the {focus} type/species. "
        "Return ONLY valid JSON with keys: species (string), confidence (0-1 float). "
        "Example: {\"species\": \"Vetiver Grass\", \"confidence\": 0.85}"
    )
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 100
    }

    for attempt in range(2):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code != 200:
                print(f"Groq vision API error: {response.status_code} - {response.text[:200]}")
                time.sleep(0.5)
                continue
                
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            if content.startswith("```"):
                content = content.strip("`")
                if content.lower().startswith("json"):
                    content = content[4:].strip()
            
            # Try to extract JSON from the response
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                import re
                json_match = re.search(r'\{[^}]+\}', content)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    continue
                    
            species = str(data.get("species", "Unknown")).strip()
            confidence = float(data.get("confidence", 0.0))
            return {"species": species, "confidence": _clamp(confidence, 0.0, 1.0)}
        except (TimeoutError, Exception) as e:
            print(f"Groq vision error: {e}")
            time.sleep(0.5)
            continue

    return None


def classify_plant_species(image_path, fast=False):
    if not fast:
        llm_result = _llm_classify_species(image_path, focus="plant")
        if llm_result:
            return {
                "species": llm_result.get("species", "Unknown"),
                "confidence": round(llm_result.get("confidence", 0.0), 2)
            }

    image = cv2.imread(image_path)
    if image is None:
        return {"species": "Unknown", "confidence": 0.0}

    if fast:
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    green_mask = (h_ch >= 35) & (h_ch <= 85) & (s_ch > 40) & (v_ch > 40)
    green_ratio = float(np.sum(green_mask) / green_mask.size)

    green_mask_u8 = (green_mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    green_mask_u8 = cv2.morphologyEx(green_mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    green_mask_u8 = cv2.morphologyEx(green_mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    contour_data = cv2.findContours(green_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contour_data[0] if len(contour_data) == 2 else contour_data[1]

    aspect_ratio = 1.0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = (h / w) if w else 1.0

    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 80, 160)
    edge_density = float(np.mean(edges > 0))

    if green_ratio < 0.04:
        return {"species": "Unknown", "confidence": 0.1}

    vetiver_score = 0.0
    if aspect_ratio > 2.5:
        vetiver_score += 0.45
    if green_ratio > 0.12:
        vetiver_score += 0.35
    if edge_density > 0.04:
        vetiver_score += 0.15

    vetiver_score = _clamp(vetiver_score, 0.0, 0.9)

    if vetiver_score >= 0.55:
        return {"species": "Vetiver", "confidence": round(vetiver_score, 2)}

    generic_score = _clamp(0.35 + green_ratio * 0.9, 0.35, 0.85)
    return {"species": "Other Plant", "confidence": round(generic_score, 2)}


def classify_root_species(image_path, fast=False):
    if not fast:
        llm_result = _llm_classify_species(image_path, focus="root")
        if llm_result:
            return {
                "species": llm_result.get("species", "Unknown"),
                "confidence": round(llm_result.get("confidence", 0.0), 2)
            }

    return {"species": "Unknown", "confidence": 0.2}
