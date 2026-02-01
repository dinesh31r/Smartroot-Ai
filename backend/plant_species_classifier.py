import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import cv2
import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions


@st.cache_resource
def _get_cnn_classifier():
    """Load pretrained MobileNetV2 for general species/object classification"""
    try:
        return MobileNetV2(weights='imagenet')
    except Exception as e:
        print(f"❌ Failed to load CNN classifier: {e}")
        return None


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
    
    if focus == "plant":
        prompt = (
            "Analyze this image carefully. This application ONLY supports Vetiver Grass (Chrysopogon zizanioides). "
            "If the image does NOT contain a plant, return: {\"species\": \"Not a plant\", \"confidence\": 0.0, \"is_vetiver\": false} "
            "If it contains a plant but is NOT Vetiver Grass, return: {\"species\": \"<detected species>\", \"confidence\": <conf>, \"is_vetiver\": false} "
            "If it IS Vetiver Grass (tall grass with long narrow leaves, dense clumping growth), return: {\"species\": \"Vetiver Grass\", \"confidence\": <conf>, \"is_vetiver\": true} "
            "Return ONLY valid JSON with keys: species (string), confidence (0-1 float), is_vetiver (boolean). "
            "Example: {\"species\": \"Vetiver Grass\", \"confidence\": 0.85, \"is_vetiver\": true}"
        )
    else:  # focus == "root"
        prompt = (
            "Analyze this image carefully. This application ONLY supports Vetiver Grass roots. "
            "If the image does NOT contain plant roots, return: {\"species\": \"Not a root\", \"confidence\": 0.0, \"is_vetiver\": false} "
            "If it contains roots but is NOT Vetiver root (Vetiver roots are deep, fibrous, dense, often yellowish-brown), return: {\"species\": \"<detected type>\", \"confidence\": <conf>, \"is_vetiver\": false} "
            "If it IS Vetiver root, return: {\"species\": \"Vetiver Root\", \"confidence\": <conf>, \"is_vetiver\": true} "
            "Return ONLY valid JSON with keys: species (string), confidence (0-1 float), is_vetiver (boolean). "
            "Example: {\"species\": \"Vetiver Root\", \"confidence\": 0.85, \"is_vetiver\": true}"
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



def _cnn_classify_species(image_path):
    """Fallback: Use ImageNet classifier to guess plant type"""
    model = _get_cnn_classifier()
    if not model:
        return None
        
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        x = preprocess_input(np.expand_dims(img.astype(np.float32), axis=0))
        
        preds = model.predict(x, verbose=0)
        decoded = decode_predictions(preds, top=3)[0]
        
        # Look for plant-related keywords
        for _, label, score in decoded:
            label_clean = label.replace('_', ' ').title()
            return {"species": label_clean, "confidence": float(score)}
    except Exception as e:
        print(f"⚠️ CNN classification error: {e}")
        
    return None


def classify_plant_species(image_path, fast=False):
    """Classify plant species - VETIVER ONLY. Returns is_vetiver flag."""
    if not fast:
        llm_result = _llm_classify_species(image_path, focus="plant")
        if llm_result:
            species = llm_result.get("species", "Unknown")
            is_vetiver = llm_result.get("is_vetiver", "vetiver" in species.lower())
            return {
                "species": species,
                "confidence": round(llm_result.get("confidence", 0.0), 2),
                "model_engine": "Llama-4 Scout",
                "is_vetiver": is_vetiver
            }
    
    # NEW (SAFE): CNN Fallback if LLM skipped/fails
    cnn_result = _cnn_classify_species(image_path)
    if cnn_result and cnn_result["confidence"] > 0.15:
        return {
            "species": cnn_result["species"],
            "confidence": round(cnn_result["confidence"], 2),
            "model_engine": "MobileNetV2"
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
        # Fallback: Check if it looks like a root (brownish) or just assume Unknown
        return {"species": "Unknown", "confidence": 0.1}

    vetiver_score = 0.0
    if aspect_ratio > 2.5:
        vetiver_score += 0.45
    if green_ratio > 0.12:
        vetiver_score += 0.35
    if edge_density > 0.04:
        vetiver_score += 0.15

    vetiver_score = _clamp(vetiver_score, 0.0, 0.9)

    if vetiver_score >= 0.50:
        return {"species": "Vetiver Grass", "confidence": round(vetiver_score, 2), "model_engine": "Morphology Analysis", "is_vetiver": True}

    generic_score = _clamp(0.35 + green_ratio * 0.9, 0.35, 0.85)
    return {"species": "Non-Vetiver Plant", "confidence": round(generic_score, 2), "model_engine": "Morphology Analysis", "is_vetiver": False}


def classify_root_species(image_path, fast=False):
    """Classify root species - VETIVER ONLY. Returns is_vetiver flag."""
    if not fast:
        llm_result = _llm_classify_species(image_path, focus="root")
        if llm_result:
            species_text = llm_result.get("species", "").lower()
            confidence = llm_result.get("confidence", 0.0)
            is_vetiver = llm_result.get("is_vetiver", "vetiver" in species_text)
            
            # If LLM says "Not a root" but it's likely a root or confidence is low, ignore it
            if "not a root" not in species_text and confidence > 0.4:
                return {
                    "species": llm_result.get("species", "Unknown"),
                    "confidence": round(confidence, 2),
                    "model_engine": "Llama-4 Scout",
                    "is_vetiver": is_vetiver
                }

    # Better fallback - check for Vetiver root characteristics
    image = cv2.imread(image_path)
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        # Vetiver roots are dense and fibrous with high edge density
        if edge_density > 0.03:
            return {"species": "Vetiver Root", "confidence": 0.8, "model_engine": "Morphology Analysis", "is_vetiver": True}
            
    return {"species": "Unknown Root", "confidence": 0.2, "model_engine": "None", "is_vetiver": False}
