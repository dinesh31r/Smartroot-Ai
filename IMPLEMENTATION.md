# SmartRoot-AI Implementation Guide

> Documentation of core working features

---

## Table of Contents

1. [Project Setup](#1-project-setup)
2. [CNN Stress Classification](#2-cnn-stress-classification)
3. [Root Image Analysis](#3-root-image-analysis)
4. [Root Growth Simulation](#4-root-growth-simulation)
5. [LLM Integration (Groq)](#5-llm-integration-groq)
6. [Species Classification](#6-species-classification)
7. [Database (Supabase)](#7-database-supabase)
8. [Deployment](#8-deployment)

---

## 1. Project Setup

### Virtual Environment
```bash
python3 -m venv venv_tf
source venv_tf/bin/activate
```

### Dependencies
```
streamlit==1.32.0
numpy==1.26.4
matplotlib==3.8.3
pillow==10.2.0
opencv-python-headless==4.9.0.80
python-dotenv
fpdf2==2.7.9
plotly==5.18.0
supabase==2.0.0
tensorflow
requests
```

### Project Structure
```
smartroot_ai/
├── app.py                 # Main Streamlit app
├── backend/
│   ├── cnn_inference.py   # CNN prediction
│   ├── root_simulator.py  # Growth simulation
│   ├── ai_realism.py      # Realism scoring
│   ├── llm_guidence.py    # Groq LLM calls
│   ├── database.py        # Supabase client
│   ├── plant_species_classifier.py
│   ├── root_image_analyzer.py
│   └── root_traits_extractor.py
├── model/
│   └── vetiver_cnn.h5     # Trained CNN model
├── static/                # CSS files
└── .streamlit/
    ├── config.toml
    └── secrets.toml
```

---

## 2. CNN Stress Classification

**File**: `backend/cnn_inference.py`

Classifies Vetiver plant stress from images using a trained CNN model.

**Classes**: Healthy, Low Moisture, Low Nutrient, Stressed

```python
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("model/vetiver_cnn.h5")

def predict_stress(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    predictions = model.predict(np.expand_dims(img_array, 0), verbose=0)[0]
    
    labels = ["Healthy", "Low Moisture", "Low Nutrient", "Stressed"]
    class_idx = np.argmax(predictions)
    
    return labels[class_idx], moisture, nutrient, confidence, weak_signal
```

---

## 3. Root Image Analysis

**File**: `backend/root_image_analyzer.py`

Extracts morphological traits from root images using OpenCV.

**Traits extracted**:
- Root density (% coverage)
- Branching count
- Average diameter
- Max depth/width

```python
import cv2
import numpy as np

def analyze_root_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    density = (np.sum(binary > 0) / binary.size) * 100
    return {"density": density, "branching_count": len(contours), ...}
```

---

## 4. Root Growth Simulation

**File**: `backend/root_simulator.py`

Generates biologically realistic root growth visualizations.

**Features**:
- Soil-aware parameters (Sandy, Clay, Loamy)
- Recursive branching with depth control
- Root hair generation

```python
def simulate_root(soil_type="Loamy", moisture=50, nutrient=50):
    # Returns matplotlib figure and growth metrics
    fig, ax = plt.subplots(figsize=(10, 12), facecolor='black')
    # ... recursive growth algorithm
    return fig, metrics
```

**File**: `backend/ai_realism.py`

Scores simulation realism (0-100) based on:
- Depth appropriate for soil type
- Branching density
- Root hair distribution
- Spread ratio

---

## 5. LLM Integration (Groq)

**File**: `backend/llm_guidence.py`

Uses Groq API (free tier) with LLaMA 3.3 70B for AI explanations.

```python
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def call_groq_api(prompt, max_tokens=500):
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": max_tokens
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]
```

**Functions**:
- `llm_explain_plant_analysis()` - Plant health explanations
- `llm_explain_root_analysis()` - Root analysis explanations  
- `llm_biological_analysis()` - Biological insights

**Fallback**: Rule-based explanations when API unavailable.

---

## 6. Species Classification

**File**: `backend/plant_species_classifier.py`

Uses Groq Vision (LLaMA 4 Scout) for species identification.

```python
def classify_plant_species(image_path):
    image_b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Identify the plant species..."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }]
    }
    # Returns {"species": "Name", "confidence": 0.85}
```

**Fallback**: OpenCV-based Vetiver detection using green color analysis and aspect ratio.

---

## 7. Database (Supabase)

**File**: `backend/database.py`

PostgreSQL database for persistent storage.

**Tables**:
- `dashboard_stats` - Aggregate statistics
- `analysis_history` - Analysis records
- `health_metrics` - Time-series health data

```python
from supabase import create_client

def get_supabase_client():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

def save_analysis_to_db(analysis_type, health_score, moisture, nutrient):
    supabase = get_supabase_client()
    supabase.table('analysis_history').insert({...}).execute()
```

**Note**: Use `supabase==2.0.0` to avoid proxy argument error.

---

## 8. Deployment

### Streamlit Config

**`.streamlit/config.toml`**:
```toml
[theme]
primaryColor = "#30d158"
backgroundColor = "#000000"
secondaryBackgroundColor = "#1c1c1e"
textColor = "#f5f5f7"

[server]
headless = true
port = 8501
```

### Secrets

**`.streamlit/secrets.toml`**:
```toml
[supabase]
url = "https://your-project.supabase.co"
key = "your_anon_key"

[groq]
api_key = "gsk_your_key"
```

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| ML Model | TensorFlow CNN |
| Vision AI | Groq LLaMA 4 Scout |
| Text AI | Groq LLaMA 3.3 70B |
| Database | Supabase PostgreSQL |
| Image Processing | OpenCV |
| Charts | Plotly + Matplotlib |

---

*Author: R. Dinesh | January 2026*
