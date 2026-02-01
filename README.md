<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?style=for-the-badge&logo=supabase" alt="Supabase">
  <img src="https://img.shields.io/badge/Groq-LLaMA%204-00D4AA?style=for-the-badge" alt="Groq">
</p>

---

# 🌱 SmartRoot-AI (Pro Edition)

### Intelligent Plant Stress Detection & Root Architecture Analysis

> A next-generation agricultural AI platform combining Ensemble Deep Learning, Computer Vision, and Sensor Fusion for precision farming.

---

## 🎯 Project Overview

**SmartRoot-AI** is a comprehensive diagnostic tool for Vetiver grass and other crops. It goes beyond simple image classification by integrating **multimodal data** (visual + environmental) and employing **ensemble AI models** to deliver highly accurate stress assessments.

### Key Capabilities
-   **Multi-Model Analysis**: Fuses predictions from Custom CNNs, ResNet50, and EfficientNetV2.
-   **Sensor Fusion**: Combines image data with temp/humidity/soil sensors for robust decision-making.
-   **Root Intelligence**: Advanced segmentation (U-Net & RootNav 2.0) for structural trait extraction.
-   **Generative AI Insights**: Uses **LLaMA 3.3 & Llama 4 Scout** (via Groq) for biological explanations.
-   **Pro UI/UX**: An "Apple-style" dark mode interface with 3D visualizations and glassmorphism.

---

## ✨ Features

### 🌿 Advanced Stress Detection
-   **Ensemble Prediction**: Averages multiple neural networks for max confidence.
-   **PlantVillage Logic**: Backup color-based analysis (Chlorosis/Necrosis algorithms).
-   **Sensor Fusion**: Overrides visual false positives if soil moisture sensors indicate stress.
-   **Explainable AI**: Grad-CAM heatmaps show exactly *where* the model is looking.

### 🧬 Root Architecture Analysis V2
-   **Dual-Engine segmentation**: 
    -   **U-Net Deep Learning**: Precise pixel-wise segmentation.
    -   **RootNav Logic**: Robust structural analysis (Depth, Width, Branching).
-   **3D Visualization**: Interactive 3D root models powered by Plotly.
-   **Traits Extracted**: Root density, total length, branching angles, convex hull area.

### 🤖 Generative AI (LLM)
-   **Species Identification**: Hybrid classification using **Llama 4 Scout (Vision)** and MobileNetV2.
-   **Biological Assistant**: Ask questions about plant physiology and get scientific answers.
-   **Smart Explanations**: "Why is my plant stressed?" answered by AI based on visual evidence.

### 📊 Professional Dashboard
-   **Real-time Analytics**: Historical trends for Moisture, Nutrient, and Health Scores.
-   **Global Stats**: Community-wide metrics stored in Supabase.
-   **Glassmorphism UI**: Modern, responsive, mobile-first design system.
-   **PDF Reports**: Export comprehensive medical-grade reports for your crops.

---

## 🏗️ Project Architecture

```
smartroot_ai/
├── app.py                      # Main Application (Streamlit)
├── backend/                    # Core Logic Modules
│   ├── cnn_inference.py        # Ensemble Integration & Sensor Fusion
│   ├── root_unet_segmentation.py # U-Net AI Segmentation
│   ├── rootnav_logic.py        # RootNav 2.0 Algorithms
│   ├── plantvillage_logic.py   # Heuristic Health Logic (Color)
│   ├── plant_species_classifier.py # Hybrid Llama 4/MobileNet ID
│   └── llm_guidence.py         # Groq API Gateway
├── model/                      # ML Models
│   ├── vetiver_cnn.h5          # Custom Lightweight CNN
│   ├── root_unet.h5            # (Optional) Root Segmentation Weights
│   └── mobilenet_plantvillage.h5 # (Optional) Transfer Learning Weights
├── static/                     # Assets & Theming
│   ├── advanced_style_enhanced.css # Pro Dark Theme
│   └── advanced_ui_interactions.js # JS Animations
└── .streamlit/                 # Configuration
    └── secrets.toml            # API Keys
```

---

## 🚀 Quick Start

### 1. Prerequisites
-   Python 3.10+
-   PIP & Virtual Environment

### 2. Installation

```bash
# Clone
git clone https://github.com/yourusername/smartroot_ai.git
cd smartroot_ai

# Environment
python3 -m venv venv_tf
source venv_tf/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

### 3. Configuration (.streamlit/secrets.toml)

```toml
[supabase]
url = "YOUR_SUPABASE_URL"
key = "YOUR_SUPABASE_ANON_KEY"

[groq]
api_key = "gsk_YOUR_GROQ_API_KEY"
```

### 4. Database Setup (Supabase SQL)

Run the provided SQL script to create tables:
-   `dashboard_stats`
-   `analysis_history`
-   `health_metrics`

*(See IMPLEMENTATION.md for full SQL schema)*

### 5. Running the App

```bash
streamlit run app.py
```
**Access**: `http://localhost:8501`

---

## ⚡ Performance

| Feature | Model/Engine | Speed | Accuracy |
| :--- | :--- | :--- | :--- |
| **Plant ID** | Llama 4 Scout | ~1.5s | ⭐⭐⭐⭐⭐ (High) |
| **Stress (Fast)** | Custom CNN | 58ms | ⭐⭐⭐ (Good) |
| **Stress (Pro)** | Ensemble (ResNet+EffNet) | 270ms | ⭐⭐⭐⭐⭐ (Elite) |
| **Root Seg** | U-Net / RootNav | 120ms | ⭐⭐⭐⭐ (Very Good) |
| **LLM Chat** | LLaMA 3.3 70B | ~0.8s | ⭐⭐⭐⭐⭐ (Excellent) |

---

## 📄 Documentation

For deep technical details, algorithms, and code logic:
👉 **[Read IMPLEMENTATION.md](IMPLEMENTATION.md)**

---

## ⚠️ Limitations
-   **Ensemble Mode**: Requires more RAM; may be slower on free cloud tiers.
-   **U-Net**: Requires `root_unet.h5` weights (fallback to RootNav if missing).
-   **LLM**: Rate limited by Groq Free Tier (30 req/min).

---

## 👤 Author
**R. Dinesh**  
MCA – AI & Data Science
