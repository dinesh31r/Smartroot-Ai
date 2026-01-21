<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?style=for-the-badge&logo=supabase" alt="Supabase">
  <img src="https://img.shields.io/badge/Groq-LLM-00D4AA?style=for-the-badge" alt="Groq">
</p>

---

# 🌱 SmartRoot-AI

### CNN-Based Plant Stress Detection with AI-Enhanced Root Architecture Analysis

> An intelligent agricultural analysis platform combining deep learning, computer vision, and LLM-powered insights for comprehensive plant and root health assessment.

---

## 🎯 Project Overview

**SmartRoot-AI** is a full-stack AI application that analyzes **Vetiver plant images** using a custom CNN model to estimate **moisture** and **nutrient** stress levels. It features:

- **Plant Stress Analysis** - CNN-based classification with confidence scoring
- **Root Image Intelligence** - Trait extraction and health assessment
- **Root Growth Simulation** - Soil-aware biologically realistic simulations
- **LLM-Powered Insights** - AI-generated explanations using Groq API
- **Persistent Database** - Supabase PostgreSQL for cloud data storage
- **Apple Dark Theme UI** - Modern, responsive Streamlit interface

---

## ✨ Key Features

### 🌿 Plant Stress Analysis
- **CNN Model**: Custom trained on Vetiver plant dataset
- **Stress Classification**: Healthy / Low Moisture / Low Nutrient / Stressed
- **Metrics**: Moisture (%), Nutrient (%), Confidence Score
- **Health Score**: Comprehensive 0-100 plant health rating

### 🧬 Root Image Intelligence
- **Trait Extraction**: Density, branching angles, diameter statistics
- **Health Assessment**: Root health index with efficiency metrics
- **Species Identification**: AI-powered plant/root species detection
- **Water & Nutrient Efficiency**: Resource uptake analysis

### 🌍 Root Growth Simulation
- **Soil-Aware Modeling**: Sandy, Clay, Loamy soil profiles
- **Biologically Realistic**: Gravity, resistance, branching patterns
- **Realism Scoring**: AI-evaluated structural authenticity
- **Downloadable Output**: High-resolution PNG exports

### 🤖 AI-Powered Insights
- **Groq LLM Integration**: Free-tier LLaMA 3.3 70B model
- **Vision Analysis**: LLaMA 4 Scout for species identification
- **Contextual Explanations**: Plant-specific care recommendations
- **Biological Reasoning**: Scientific analysis of root structures

### 💾 Cloud Database
- **Supabase PostgreSQL**: Persistent data storage
- **Dashboard Stats**: Analysis counts, health averages
- **Analysis History**: Timestamped analysis records
- **Health Metrics**: Trend tracking over time

### 🎨 Modern UI/UX
- **Apple Dark Theme**: Pure black (#000000) aesthetic
- **Responsive Design**: Mobile-friendly layout
- **Smooth Animations**: CSS transitions and effects
- **PDF Reports**: Downloadable analysis summaries

---

## 🏗️ Project Architecture

```
smartroot_ai/
├── app.py                          # Main Streamlit application (2100+ lines)
├── backend/
│   ├── __init__.py
│   ├── ai_realism.py               # Root structure realism evaluation
│   ├── cnn_inference.py            # CNN model inference
│   ├── cnn_model_arch.py           # Model architecture definition
│   ├── database.py                 # Supabase PostgreSQL integration
│   ├── llm_guidence.py             # Groq LLM API integration
│   ├── plant_species_classifier.py # Vision-based species detection
│   ├── root_analytics.py           # Root metric calculations
│   ├── root_health_classifier.py   # Health classification logic
│   ├── root_image_analyzer.py      # OpenCV image processing
│   ├── root_simulator.py           # Root growth simulation
│   ├── root_soil_inference.py      # Soil type inference
│   └── root_traits_extractor.py    # Trait extraction algorithms
├── model/
│   └── vetiver_cnn.h5              # Trained CNN weights
├── static/
│   ├── advanced_style_enhanced.css # Main Apple dark theme
│   ├── advanced_ui_animations.css  # Animation effects
│   ├── dashboard_components.css    # Dashboard styling
│   └── logo_animated.css           # Logo animations
├── .streamlit/
│   ├── config.toml                 # Streamlit configuration
│   └── secrets.toml                # API keys & credentials
├── requirements.txt                # Python dependencies
├── IMPLEMENTATION.md               # Detailed implementation guide
├── Dockerfile                      # Container configuration
└── runtime.txt                     # Python version specification
```

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|------------|
| **Frontend** | Streamlit 1.32.0, Custom CSS, HTML/JS |
| **Backend** | Python 3.10, TensorFlow/Keras |
| **Computer Vision** | OpenCV, NumPy, PIL |
| **LLM** | Groq API (LLaMA 3.3 70B, LLaMA 4 Scout) |
| **Database** | Supabase PostgreSQL |
| **Visualization** | Matplotlib, Plotly |
| **Deployment** | Docker, Streamlit Cloud |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip package manager

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/smartroot_ai.git
cd smartroot_ai

# Create virtual environment
python3 -m venv venv_tf
source venv_tf/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create `.streamlit/secrets.toml`:

```toml
# Supabase Configuration
[supabase]
url = "your_supabase_url"
key = "your_supabase_anon_key"

# Groq API Configuration (FREE)
# Get key at: https://console.groq.com/keys
[groq]
api_key = "gsk_your_groq_api_key"
```

### Database Setup

Run in Supabase SQL Editor:

```sql
-- Dashboard Stats
CREATE TABLE dashboard_stats (
    id SERIAL PRIMARY KEY,
    total_plants INTEGER DEFAULT 0,
    total_roots INTEGER DEFAULT 0,
    total_simulations INTEGER DEFAULT 0,
    avg_health_score FLOAT DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Analysis History
CREATE TABLE analysis_history (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    icon VARCHAR(10) NOT NULL,
    health_score INTEGER NOT NULL,
    analysis_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Health Metrics
CREATE TABLE health_metrics (
    id SERIAL PRIMARY KEY,
    health_score INTEGER,
    moisture INTEGER,
    nutrient INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Initial data
INSERT INTO dashboard_stats (id, total_plants, total_roots, total_simulations, avg_health_score)
VALUES (1, 0, 0, 0, 0) ON CONFLICT (id) DO NOTHING;
```

### Run Application

```bash
streamlit run app.py --server.port 8501
```

Open: `http://localhost:8501`

---

## ☁️ Streamlit Cloud Deployment

### Step 1: Prepare Repository

1. Push your code to GitHub:
```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

2. Ensure these files are included:
   - `app.py` - Main application
   - `requirements.txt` - Dependencies
   - `model/vetiver_cnn.h5` - CNN model weights
   - `.streamlit/config.toml` - Theme configuration
   - `backend/` - All backend modules
   - `static/` - CSS files

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub repository
4. Set:
   - **Repository**: `your-username/smartroot_ai`
   - **Branch**: `main`
   - **Main file path**: `app.py`

### Step 3: Configure Secrets

In Streamlit Cloud dashboard → **Settings** → **Secrets**:

```toml
[supabase]
url = "https://your-project.supabase.co"
key = "your_anon_key"

[groq]
api_key = "gsk_your_api_key"
```

### Step 4: Deploy

Click **"Deploy!"** - Your app will be live at:
`https://your-app-name.streamlit.app`

### Mobile Access

The app is fully responsive and works on:
- 📱 Smartphones (iOS/Android)
- 📱 Tablets
- 💻 Desktops

---

## 📊 API Integrations

### Groq API (Free Tier)
- **Model**: `llama-3.3-70b-versatile` for text analysis
- **Vision**: `meta-llama/llama-4-scout-17b-16e-instruct` for image analysis
- **Limits**: 30 requests/minute, 14,400 requests/day
- **Cost**: FREE

### Supabase (Free Tier)
- **Database**: PostgreSQL with 500MB storage
- **API**: REST and real-time subscriptions
- **Auth**: Row-level security support
- **Cost**: FREE

---

## 📱 Usage Guide

1. **Upload Plant Image** - Select a Vetiver plant photo (JPG/PNG)
2. **Configure Settings** - Choose soil type and analysis mode
3. **View Results** - See stress analysis, health scores, AI insights
4. **Upload Root Image** (Optional) - Analyze root structure
5. **Download Reports** - Export PDF analysis summaries

---

## 🎨 UI Theme

The application features a custom **Apple Dark Theme**:
- Pure black background (#000000)
- Apple SF-style typography
- Green accent color (#30d158)
- Borderless, minimal design
- Smooth hover transitions

---

## 📄 Documentation

- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Detailed implementation guide
- **[requirements.txt](requirements.txt)** - Dependency list

---

## ⚠️ Known Limitations

- TensorFlow requires significant memory
- CNN accuracy depends on training dataset quality
- LLM responses may vary based on prompt
- Free tier APIs have rate limits

---

## 🔮 Future Enhancements

- [ ] Multi-plant species support
- [ ] Time-series root growth animation
- [ ] PlantCV integration
- [ ] Mobile native app
- [ ] Real-time monitoring dashboard

---

## 🧑‍🎓 Academic Use

Suitable for:
- MCA/MTech Minor/Major Projects
- AI/ML Research Projects
- Agricultural Technology Studies
- Computer Vision Coursework

---

## 📜 License

This project is developed for **educational and research purposes**.

---

## 👤 Author

**R. Dinesh**  
MCA – AI & Data Science

---

<p align="center">
  Made with ❤️ using Python, TensorFlow, and AI
</p>


