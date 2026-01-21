"""
SmartRoot-AI
CNN-based Stress Detection with AI-Enhanced Root Architecture Visualization
"""

import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import math
import io
import hashlib
import json
import random
import warnings
from fpdf import FPDF
import os
from PIL import Image
import textwrap
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from backend.llm_guidence import llm_biological_analysis, llm_health_check
from backend.cnn_inference import predict_stress
from backend.root_simulator import simulate_root
from backend.ai_realism import evaluate_root_realism
from backend.database import (
    get_supabase_client,
    sync_stats_to_session,
    save_analysis_to_db,
    get_dashboard_stats,
    get_analysis_history,
    get_health_metrics
)
from backend.plant_species_classifier import classify_plant_species, classify_root_species
from backend.root_image_analyzer import analyze_root_image
from backend.llm_guidence import llm_explain_plant_analysis, llm_explain_root_analysis

warnings.filterwarnings(
    "ignore",
    message="coroutine 'expire_cache' was never awaited",
    category=RuntimeWarning
)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="SmartRoot-AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Initialize session state for analysis history
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Initialize session state for dashboard stats
if 'dashboard_stats' not in st.session_state:
    st.session_state.dashboard_stats = {
        'total_plants': 0,
        'total_roots': 0,
        'total_simulations': 0,
        'avg_health_score': 0,
        'health_scores': [],
        'moisture_history': [],
        'nutrient_history': [],
        'timestamps': []
    }

# Initialize session state for current analysis step
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# Sync with Supabase database on first load
if 'db_synced' not in st.session_state:
    try:
        sync_stats_to_session()
        st.session_state.db_synced = True
    except Exception as e:
        st.session_state.db_synced = False
        print(f"Database sync skipped: {e}")

# -------------------------------------------------
# QUERY PARAM HELPERS (MOBILE SAFE MODE)
# -------------------------------------------------
def _get_query_param(key, default=None):
    try:
        params = st.query_params
        value = params.get(key, default)
        if isinstance(value, list):
            return value[0] if value else default
        return value if value is not None else default
    except Exception:
        try:
            params = st.experimental_get_query_params()
            value = params.get(key, [default])
            return value[0] if isinstance(value, list) else value
        except Exception:
            return default

MOBILE_SAFE_MODE = str(_get_query_param("mobile", "0")).lower() in {"1", "true", "yes"}

# -------------------------------------------------
# CACHED STATIC ASSET LOADER
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# -------------------------------------------------
# LOAD CSS & ADVANCED UI ENHANCEMENTS
# -------------------------------------------------

if not MOBILE_SAFE_MODE:
        st.markdown(f"<style>{_load_text('static/advanced_style_enhanced.css')}</style>", unsafe_allow_html=True)

        st.markdown(f"<style>{_load_text('static/advanced_ui_animations.css')}</style>", unsafe_allow_html=True)

        # Load new animated logo CSS
        st.markdown(f"<style>{_load_text('static/logo_animated.css')}</style>", unsafe_allow_html=True)

        # Load dashboard components CSS
        st.markdown(f"<style>{_load_text('static/dashboard_components.css')}</style>", unsafe_allow_html=True)

        # Apple Dark Theme - Clean File Uploader Styling
        st.markdown(
                """
                <style>
                    /* ===== APPLE DARK THEME FILE UPLOADER ===== */
                    
                    /* Remove all backgrounds and borders from file uploader container */
                    [data-testid="stFileUploader"],
                    [data-testid="stFileUploader"] > div,
                    [data-testid="stFileUploader"] > label,
                    [data-testid="stFileUploader"] > label > div,
                    [data-testid="stFileUploader"] [data-testid="stWidgetLabel"],
                    [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] > div,
                    .stFileUploader,
                    .stFileUploader > div,
                    .stFileUploader > label {
                        background: transparent !important;
                        background-color: transparent !important;
                        border: none !important;
                        box-shadow: none !important;
                    }
                    
                    /* Label text - clean white */
                    [data-testid="stFileUploader"] label p,
                    [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p,
                    [data-testid="stWidgetLabel"] p {
                        color: #f5f5f7 !important;
                        font-weight: 500 !important;
                        font-size: 0.95rem !important;
                        background: transparent !important;
                    }
                    
                    /* Dropzone - Pure black, no border */
                    [data-testid="stFileUploader"] section,
                    [data-testid="stFileUploaderDropzone"],
                    [data-testid="stFileUploader"] > div > section,
                    [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"],
                    .stFileUploader section {
                        background: #000000 !important;
                        border: none !important;
                        border-radius: 12px !important;
                        padding: 1.25rem !important;
                    }
                    
                    [data-testid="stFileUploader"] section:hover {
                        background: #1a1a1a !important;
                    }
                    
                    /* Helper text */
                    [data-testid="stFileUploader"] small {
                        color: #6e6e73 !important;
                    }
                    
                    /* Drag and drop text */
                    [data-testid="stFileUploader"] section span,
                    [data-testid="stFileUploader"] section div {
                        color: #86868b !important;
                    }
                    
                    /* Browse button - Apple green */
                    [data-testid="stFileUploader"] button,
                    [data-testid="baseButton-secondary"] {
                        background: linear-gradient(135deg, #30d158, #28a745) !important;
                        color: #ffffff !important;
                        border: none !important;
                        font-weight: 600 !important;
                        padding: 0.625rem 1.25rem !important;
                        border-radius: 12px !important;
                    }
                    
                    [data-testid="stFileUploader"] button:hover {
                        transform: scale(1.02) !important;
                        box-shadow: 0 4px 12px rgba(48,209,88,0.4) !important;
                    }
                    
                    [data-testid="stFileUploader"] button *,
                    [data-testid="baseButton-secondary"] * {
                        color: #ffffff !important;
                    }
                    
                    /* Uploaded file styling */
                    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
                        background: #1c1c1e !important;
                        border-radius: 8px !important;
                        border: none !important;
                    }
                    
                    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
                        color: #f5f5f7 !important;
                    }
                    
                    /* Remove any remaining borders */
                    [data-testid="stFileUploader"] *,
                    .stFileUploader * {
                        border-color: transparent !important;
                    }
                </style>
                """,
                unsafe_allow_html=True
        )

        # Custom header with Apple-like minimalism (using external CSS)
        header_html = textwrap.dedent("""
            <div style="padding: 4rem 1rem 2rem 1rem; text-align: center;">
                <div class="logo-container" style="margin-bottom: 1.5rem;">
                    <svg style="width: 120px; height: 120px; display: block; margin: 0 auto;" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                        <!-- Simplified Logo -->
                        <path d="M 50 85 Q 45 75 45 65 Q 45 55 50 50" stroke="#34c759" stroke-width="3" fill="none" stroke-linecap="round"/>
                        <path d="M 45 65 L 35 75" stroke="#34c759" stroke-width="2.5" fill="none" stroke-linecap="round"/>
                        <path d="M 50 50 Q 52 35 50 20" stroke="#34c759" stroke-width="3" fill="none" stroke-linecap="round"/>
                        <ellipse cx="40" cy="35" rx="8" ry="12" fill="#34c759" opacity="0.8" transform="rotate(-35 40 35)"/>
                        <ellipse cx="60" cy="38" rx="8" ry="12" fill="#34c759" opacity="0.8" transform="rotate(35 60 38)"/>
                    </svg>
                </div>
                <h1 style="margin: 0; font-size: 3.5rem; letter-spacing: -0.03em;">SmartRoot AI</h1>
                <p style="margin-top: 1rem; color: #86868b; font-size: 1.25rem;">Advanced root analysis powered by intelligence.</p>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; margin-top: 3rem; max-width: 800px; margin-left: auto; margin-right: auto;">
                    <div class="feature-card">
                        <div class="feature-icon">📷</div>
                        <p>Upload</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">📊</div>
                        <p>Traits</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">🩺</div>
                        <p>Health</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">🧠</div>
                        <p>Stress</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">📈</div>
                        <p>Simulate</p>
                    </div>
                </div>
            </div>
            """)
        header_html = "\n".join(line.lstrip() for line in header_html.splitlines()).strip()
        st.markdown(header_html, unsafe_allow_html=True)
else:
        st.markdown(
                """
                <style>
                    html, body, .stApp, .main {
                        background: #ffffff !important;
                        color: #111827 !important;
                    }
                    .block-container {
                        padding: 1.25rem 1rem !important;
                    }
                </style>
                <div style="padding: 1rem 0;">
                    <h2 style="margin: 0 0 0.5rem 0; color: #111827;">SmartRoot-AI</h2>
                    <p style="margin: 0; color: #4b5563;">Mobile safe mode enabled for better compatibility.</p>
                </div>
                """,
                unsafe_allow_html=True
        )

# -------------------------------------------------
# THEME SWITCHER - Lightweight implementation
# -------------------------------------------------
# Simple theme state managed via Streamlit session
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    try {
        const scrollProgress = document.getElementById('scrollProgress');
        if (scrollProgress) {
            window.addEventListener('scroll', function() {
                const scrollTop = window.scrollY;
                const docHeight = document.documentElement.scrollHeight - window.innerHeight;
                const scrolled = (scrollTop / docHeight) * 100;
                scrollProgress.style.width = scrolled + '%';
            });
        }
    } catch (e) {
        console.log('Scroll progress not available');
    }
});
</script>
""", unsafe_allow_html=True)
# No additional script needed here

# -------------------------------------------------
# LOAD ADVANCED UI INTERACTIONS SCRIPT
# -------------------------------------------------
if not MOBILE_SAFE_MODE:
    ui_js = _load_text("static/advanced_ui_interactions.js")
    st.markdown(
        f"""
        <script>
        {ui_js}
        </script>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------
# LOAD PARTICLES JS
# -------------------------------------------------
# Particle background (optional - currently disabled)
# with open("static/particles.js") as f:
#     particles_js = f.read()
#     st.markdown(
#         f"""
#         <script>
#         {particles_js}
#         </script>
#         """,
#         unsafe_allow_html=True
#     )


# -------------------------------------------------
# PROGRESS STEPPER COMPONENT
# -------------------------------------------------
def render_progress_stepper(current_step):
    """Render a visual progress stepper showing analysis workflow"""
    steps = [
        ("📤", "Upload"),
        ("🔍", "Analysis"),
        ("🌱", "Simulate"),
        ("📊", "Report")
    ]
    
    # Calculate track width based on current step
    track_width = (current_step / (len(steps) - 1)) * 80 if current_step > 0 else 0
    
    stepper_html = f"""
    <div class="progress-stepper">
        <div class="progress-stepper-track" style="width: {track_width}%;"></div>
    """
    
    for i, (icon, label) in enumerate(steps):
        if i < current_step:
            status = "completed"
            icon_display = "✓"
        elif i == current_step:
            status = "active"
            icon_display = icon
        else:
            status = "pending"
            icon_display = icon
        
        stepper_html += f"""
        <div class="step {status}">
            <div class="step-circle">{icon_display}</div>
            <div class="step-label">{label}</div>
        </div>
        """
    
    stepper_html += "</div>"
    return stepper_html


# -------------------------------------------------
# CIRCULAR HEALTH SCORE COMPONENT
# -------------------------------------------------
def render_circular_health_score(score, label="Health Score"):
    """Render an animated circular progress indicator for health scores"""
    # Determine color gradient based on score
    if score >= 80:
        gradient_class = "excellent"
        color_start = "#10b981"
        color_end = "#059669"
    elif score >= 60:
        gradient_class = "good"
        color_start = "#22c55e"
        color_end = "#16a34a"
    elif score >= 40:
        gradient_class = "moderate"
        color_start = "#f59e0b"
        color_end = "#d97706"
    else:
        gradient_class = "poor"
        color_start = "#ef4444"
        color_end = "#dc2626"
    
    # Calculate stroke-dashoffset (440 is full circle, 0 is complete)
    offset = 440 - (440 * score / 100)
    
    return f"""
    <div class="health-score-container">
        <div class="circular-progress">
            <svg viewBox="0 0 160 160">
                <defs>
                    <linearGradient id="gradient-{gradient_class}" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:{color_start}" />
                        <stop offset="100%" style="stop-color:{color_end}" />
                    </linearGradient>
                </defs>
                <circle class="progress-bg" cx="80" cy="80" r="70"></circle>
                <circle class="progress-bar {gradient_class}" cx="80" cy="80" r="70" 
                        style="stroke-dashoffset: {offset}; stroke: url(#gradient-{gradient_class});"></circle>
            </svg>
            <div class="progress-value animate">
                <div class="score-number" style="background: linear-gradient(135deg, {color_start}, {color_end}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{score}</div>
                <div class="score-label">{label}</div>
            </div>
        </div>
    </div>
    """


# -------------------------------------------------
# MINI GAUGE COMPONENT
# -------------------------------------------------
def render_mini_gauge(value, label, color="#10b981"):
    """Render a small circular gauge for individual metrics"""
    offset = 283 - (283 * value / 100)
    
    return f"""<div style="display: flex; flex-direction: column; align-items: center;">
        <div style="position: relative; width: 100px; height: 100px;">
            <svg viewBox="0 0 100 100" style="width: 100%; height: 100%; transform: rotate(-90deg);">
                <circle cx="50" cy="50" r="45" fill="none" stroke="#e5e5e5" stroke-width="8"></circle>
                <circle cx="50" cy="50" r="45" fill="none" stroke="{color}" stroke-width="8" 
                        stroke-dasharray="283" stroke-dashoffset="{offset}" stroke-linecap="round"></circle>
            </svg>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                <div style="font-size: 1.25rem; font-weight: 700; color: {color};">{value}%</div>
            </div>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #666666; font-weight: 500;">{label}</div>
    </div>"""


# -------------------------------------------------
# DASHBOARD STATS CARDS
# -------------------------------------------------
def render_dashboard_stats(stats):
    """Render the dashboard stats cards"""
    return f"""
    <div class="dashboard-grid">
        <div class="stat-card plants">
            <div class="stat-icon">🌿</div>
            <div class="stat-value animate">{stats['total_plants']}</div>
            <div class="stat-label">Plants Analyzed</div>
        </div>
        <div class="stat-card roots">
            <div class="stat-icon">🧬</div>
            <div class="stat-value animate">{stats['total_roots']}</div>
            <div class="stat-label">Roots Analyzed</div>
        </div>
        <div class="stat-card health">
            <div class="stat-icon">💚</div>
            <div class="stat-value animate">{stats['avg_health_score']:.0f}</div>
            <div class="stat-label">Avg Health Score</div>
        </div>
        <div class="stat-card simulations">
            <div class="stat-icon">📈</div>
            <div class="stat-value animate">{stats['total_simulations']}</div>
            <div class="stat-label">Simulations Run</div>
        </div>
    </div>
    """


# -------------------------------------------------
# ANALYSIS HISTORY COMPONENT
# -------------------------------------------------
def render_analysis_history(history):
    """Render the recent analysis history list"""
    if not history:
        return """
        <div class="history-container">
            <div class="history-header">
                <div class="history-title">📋 Recent Analyses</div>
            </div>
            <div class="empty-history">
                <div class="empty-history-icon">📊</div>
                <div class="empty-history-text">No analyses yet</div>
                <div class="empty-history-subtext">Upload an image to get started</div>
            </div>
        </div>
        """
    
    items_html = ""
    for item in history[-5:][::-1]:  # Show last 5, newest first
        badge_class = "healthy" if item.get('health_score', 0) >= 70 else ("stressed" if item.get('health_score', 0) >= 40 else "critical")
        badge_text = "Healthy" if item.get('health_score', 0) >= 70 else ("Stressed" if item.get('health_score', 0) >= 40 else "Critical")
        
        items_html += f"""
        <div class="history-item">
            <div class="history-thumbnail" style="display: flex; align-items: center; justify-content: center; font-size: 1.5rem;">
                {item.get('icon', '🌱')}
            </div>
            <div class="history-info">
                <div class="history-name">{item.get('name', 'Analysis')}</div>
                <div class="history-meta">
                    <span>{item.get('timestamp', '')}</span>
                    <span class="history-badge {badge_class}">{badge_text}</span>
                </div>
            </div>
            <div class="history-score">{item.get('health_score', 0)}</div>
        </div>
        """
    
    return f"""
    <div class="history-container">
        <div class="history-header">
            <div class="history-title">📋 Recent Analyses</div>
        </div>
        <div class="history-list">
            {items_html}
        </div>
    </div>
    """


# -------------------------------------------------
# INTERACTIVE PLOTLY CHARTS
# -------------------------------------------------
def create_trend_chart(stats):
    """Create an interactive trend chart using Plotly"""
    if not stats['timestamps']:
        return None
    
    fig = go.Figure()
    
    # Add moisture trace
    fig.add_trace(go.Scatter(
        x=stats['timestamps'],
        y=stats['moisture_history'],
        name='Moisture',
        line=dict(color='#0ea5e9', width=3),
        mode='lines+markers',
        marker=dict(size=8),
        hovertemplate='<b>Moisture</b>: %{y}%<extra></extra>'
    ))
    
    # Add nutrient trace
    fig.add_trace(go.Scatter(
        x=stats['timestamps'],
        y=stats['nutrient_history'],
        name='Nutrients',
        line=dict(color='#10b981', width=3),
        mode='lines+markers',
        marker=dict(size=8),
        hovertemplate='<b>Nutrients</b>: %{y}%<extra></extra>'
    ))
    
    # Add health score trace
    fig.add_trace(go.Scatter(
        x=stats['timestamps'],
        y=stats['health_scores'],
        name='Health Score',
        line=dict(color='#f59e0b', width=3),
        mode='lines+markers',
        marker=dict(size=8),
        hovertemplate='<b>Health</b>: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title="Value",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
            color='#ffffff'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#ffffff')
        ),
        margin=dict(l=40, r=20, t=40, b=20),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            range=[0, 100],
            tickfont=dict(color='#ffffff'),
            title_font=dict(color='#ffffff')
        ),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff')
        )
    )
    
    return fig


def create_root_3d_visualization(segments, max_depth):
    """Create an interactive 3D root visualization using Plotly"""
    if not segments:
        return None
    
    # Prepare data for 3D visualization
    x_coords = []
    y_coords = []
    z_coords = []
    colors = []
    widths = []
    
    xs = [r["x1"] for r in segments] + [r["x2"] for r in segments]
    center_x = np.mean(xs)
    
    for r in segments:
        depth_ratio = min(r["y2"] / max_depth, 1.0) if max_depth > 0 else 0
        
        # Create line segment
        x_coords.extend([r["x1"] - center_x, r["x2"] - center_x, None])
        y_coords.extend([r["y1"], r["y2"], None])
        # Add z variation for 3D effect
        z_offset = random.uniform(-0.3, 0.3)
        z_coords.extend([z_offset, z_offset + random.uniform(-0.1, 0.1), None])
        
        # Color based on depth
        color_val = int(180 - 120 * depth_ratio)
        colors.extend([f'rgb({color_val}, {int(color_val * 0.7)}, {int(color_val * 0.3)})', 
                       f'rgb({color_val}, {int(color_val * 0.7)}, {int(color_val * 0.3)})', None])
        widths.extend([r["thickness"] * 2, r["thickness"] * 2, None])
    
    fig = go.Figure()
    
    # Add root segments as 3D lines
    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='lines',
        line=dict(
            color='#8B4513',
            width=4
        ),
        hoverinfo='skip',
        name='Root System'
    ))
    
    # Add soil surface
    soil_x = np.linspace(min(xs) - center_x - 1, max(xs) - center_x + 1, 20)
    soil_z = np.linspace(-1, 1, 20)
    soil_x, soil_z = np.meshgrid(soil_x, soil_z)
    soil_y = np.zeros_like(soil_x)
    
    fig.add_trace(go.Surface(
        x=soil_x,
        y=soil_y,
        z=soil_z,
        colorscale=[[0, '#3b2f1e'], [1, '#5d4e37']],
        showscale=False,
        opacity=0.7,
        name='Soil Surface'
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title='', autorange='reversed'),
            zaxis=dict(showbackground=False, showticklabels=False, title=''),
            bgcolor='rgba(27, 20, 15, 1)',
            camera=dict(
                eye=dict(x=1.5, y=0.8, z=0.8)
            )
        ),
        paper_bgcolor='rgba(27, 20, 15, 1)',
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    return fig


def create_metrics_radar_chart(metrics):
    """Create a radar chart for root metrics"""
    categories = ['Depth', 'Spread', 'Density', 'Hair Count', 'Realism']
    
    # Normalize metrics to 0-100 scale
    max_vals = {'depth': 20, 'spread': 10, 'density': 100, 'hair_count': 50, 'realism': 100}
    
    values = [
        min(100, (metrics.get('max_depth_cm', 0) / max_vals['depth']) * 100),
        min(100, (metrics.get('horizontal_spread_cm', 0) / max_vals['spread']) * 100),
        min(100, (metrics.get('total_length_cm', 0) / 50) * 100),
        min(100, (metrics.get('root_hair_count', 0) / max_vals['hair_count']) * 100),
        metrics.get('realism_score', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(16, 185, 129, 0.2)',
        line=dict(color='#10b981', width=2),
        marker=dict(size=8, color='#10b981'),
        name='Root Metrics'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                tickfont=dict(size=10),
                gridcolor='rgba(0,0,0,0.1)'
            ),
            angularaxis=dict(
                gridcolor='rgba(0,0,0,0.1)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=60, r=60, t=40, b=40)
    )
    
    return fig


# -------------------------------------------------
# UPDATE DASHBOARD STATS
# -------------------------------------------------
def update_dashboard_stats(analysis_type, health_score=None, moisture=None, nutrient=None):
    """Update the dashboard statistics after an analysis"""
    stats = st.session_state.dashboard_stats
    
    if analysis_type == 'plant':
        stats['total_plants'] += 1
    elif analysis_type == 'root':
        stats['total_roots'] += 1
    elif analysis_type == 'simulation':
        stats['total_simulations'] += 1
    
    if health_score is not None:
        stats['health_scores'].append(health_score)
        stats['avg_health_score'] = sum(stats['health_scores']) / len(stats['health_scores'])
    
    if moisture is not None:
        stats['moisture_history'].append(moisture)
    
    if nutrient is not None:
        stats['nutrient_history'].append(nutrient)
    
    stats['timestamps'].append(datetime.now().strftime('%H:%M'))
    
    st.session_state.dashboard_stats = stats


def add_to_analysis_history(name, icon, health_score, analysis_type, moisture=50, nutrient=50):
    """Add an entry to the analysis history (session + database)"""
    entry = {
        'name': name,
        'icon': icon,
        'health_score': health_score,
        'analysis_type': analysis_type,
        'timestamp': datetime.now().strftime('%I:%M %p')
    }
    st.session_state.analysis_history.append(entry)
    
    # Save to Supabase database
    try:
        save_analysis_to_db(name, icon, health_score, analysis_type, moisture, nutrient)
    except Exception as e:
        print(f"Database save skipped: {e}")


# -------------------------------------------------
# AI RENDER PARAMETER ADAPTER
# -------------------------------------------------
def ai_render_params(realism_score):
    return {
        "thickness_scale": min(1.35, max(0.75, 0.95 + realism_score / 480)),
        "depth_fade": min(0.80, max(0.30, 0.5 + (100 - realism_score) / 260)),
        "jitter": max(0.006, (100 - realism_score) / 360),
        "alpha": min(0.95, max(0.40, 0.9 + realism_score / 450))
    }


def _stable_seed(*values):
    key = "|".join(str(v) for v in values)
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def _bytes_to_temp_file(file_bytes, suffix=".png"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        return tmp.name


def _cache_store():
    if "_smartroot_cache" not in st.session_state:
        st.session_state._smartroot_cache = {}
    return st.session_state._smartroot_cache


def _cache_get(key):
    return _cache_store().get(key)


def _cache_set(key, value):
    _cache_store()[key] = value
    return value


@st.cache_data(show_spinner=False)
def cached_predict_stress(file_bytes):
    key = f"stress|{hashlib.md5(file_bytes).hexdigest()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    image_path = _bytes_to_temp_file(file_bytes)
    return _cache_set(key, predict_stress(image_path))


@st.cache_data(show_spinner=False)
def cached_classify_plant_species(file_bytes):
    key = f"plant_species|{hashlib.md5(file_bytes).hexdigest()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    image_path = _bytes_to_temp_file(file_bytes)
    return _cache_set(key, classify_plant_species(image_path))


@st.cache_data(show_spinner=False)
def cached_classify_plant_species_fast(file_bytes):
    key = f"plant_species_fast|{hashlib.md5(file_bytes).hexdigest()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    image_path = _bytes_to_temp_file(file_bytes)
    return _cache_set(key, classify_plant_species(image_path, fast=True))


@st.cache_data(show_spinner=False)
def cached_classify_root_species(file_bytes):
    key = f"root_species|{hashlib.md5(file_bytes).hexdigest()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    image_path = _bytes_to_temp_file(file_bytes)
    return _cache_set(key, classify_root_species(image_path))


@st.cache_data(show_spinner=False)
def cached_classify_root_species_fast(file_bytes):
    key = f"root_species_fast|{hashlib.md5(file_bytes).hexdigest()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    image_path = _bytes_to_temp_file(file_bytes)
    return _cache_set(key, classify_root_species(image_path, fast=True))


@st.cache_data(show_spinner=False)
def cached_analyze_root_image(file_bytes):
    key = f"root_analyze|{hashlib.md5(file_bytes).hexdigest()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    image_path = _bytes_to_temp_file(file_bytes)
    return _cache_set(key, analyze_root_image(image_path))


@st.cache_data(show_spinner=False)
def cached_analyze_root_image_fast(file_bytes):
    key = f"root_analyze_fast|{hashlib.md5(file_bytes).hexdigest()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    image_path = _bytes_to_temp_file(file_bytes)
    return _cache_set(key, analyze_root_image(image_path, fast=True))



@st.cache_data(show_spinner=False)
def cached_llm_analysis(metrics, soil_type):
    payload = json.dumps(metrics, sort_keys=True, default=str)
    key = f"llm|{soil_type}|{hashlib.md5(payload.encode()).hexdigest()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    return _cache_set(key, llm_biological_analysis(metrics, soil_type))


@st.cache_data(show_spinner=False)
def cached_simulate_root(moisture, nutrient, soil_type):
    key = f"sim|{soil_type}|{round(moisture, 2)}|{round(nutrient, 2)}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    seed = _stable_seed(round(moisture, 2), round(nutrient, 2), soil_type)
    state = random.getstate()
    random.seed(seed)
    try:
        return _cache_set(key, simulate_root(moisture, nutrient, soil_type))
    finally:
        random.setstate(state)


def _decode_image_bytes(file_bytes):
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _overlay_mask(base_rgb, mask, color=(16, 185, 129), alpha=0.45):
    overlay = base_rgb.copy()
    if mask is None:
        return overlay
    mask_pixels = mask > 0
    color_arr = np.array(color, dtype=np.uint8)
    overlay[mask_pixels] = (overlay[mask_pixels] * (1 - alpha) + color_arr * alpha).astype(np.uint8)
    return overlay


def _overlay_skeleton(base_rgb, skeleton, color=(239, 68, 68)):
    overlay = base_rgb.copy()
    if skeleton is None:
        return overlay
    skel_pixels = skeleton > 0
    overlay[skel_pixels] = np.array(color, dtype=np.uint8)
    return overlay


def _colorize_heatmap(map_gray):
    if map_gray is None:
        return None
    heat_bgr = cv2.applyColorMap(map_gray.astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def _build_pdf_report(title, sections):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, _safe_pdf_text(title, max_len=60), ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", size=11)
    for heading, rows in sections:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, _safe_pdf_text(heading, max_len=60), ln=True)
        pdf.set_font("Arial", size=9)
        for key, value in rows:
            safe_key = _safe_pdf_text(key, max_len=40)
            safe_value = _safe_pdf_text(value, max_len=70)
            try:
                pdf.multi_cell(0, 5, f"- {safe_key}: {safe_value}")
            except Exception:
                pdf.multi_cell(0, 5, f"- {safe_key}: (error)")
        pdf.ln(1)
    return pdf.output(dest="S").encode("latin-1")


def _safe_pdf_text(value, max_len=100):
    """Safely convert value to PDF-compatible text with length limit"""
    text = str(value)
    # Truncate very long text to prevent horizontal overflow
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text.encode("latin-1", "replace").decode("latin-1")


def _build_root_image_report_pdf(root_report, root_species, root_image_bytes, filename_hint="root_image.png"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "SmartRoot-AI Root Image Report", ln=True)

    pdf.set_font("Arial", size=11)
    species = _safe_pdf_text(root_species.get('species', 'Unknown'), max_len=60)
    conf_text = _safe_pdf_text(f"{root_species.get('confidence', 0.0):.0%}", max_len=20)
    pdf.cell(0, 6, f"Root Species: {species}", ln=True)
    pdf.cell(0, 6, f"Species Confidence: {conf_text}", ln=True)
    pdf.ln(2)

    image_tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_root_upload.png")
    try:
        img = Image.open(io.BytesIO(root_image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(image_tmp.name, format="PNG")
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Uploaded Root Image", ln=True)
        pdf.ln(2)
        pdf.image(image_tmp.name, w=160)
        pdf.ln(4)
    except Exception as img_err:
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 6, f"(Image could not be embedded: {str(img_err)[:50]})", ln=True)
    finally:
        image_tmp.close()
        try:
            os.unlink(image_tmp.name)
        except OSError:
            pass

    def add_section(title, rows):
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, _safe_pdf_text(title, max_len=60), ln=True)
        pdf.set_font("Arial", size=9)
        for key, value in rows:
            # Use multi_cell with proper width to prevent overflow
            safe_key = _safe_pdf_text(key, max_len=40)
            safe_value = _safe_pdf_text(value, max_len=70)
            try:
                line_text = f"- {safe_key}: {safe_value}"
                if len(line_text) > 0:
                    pdf.multi_cell(0, 5, line_text)
            except Exception:
                pdf.multi_cell(0, 5, f"- {safe_key}: (error)")
        pdf.ln(1)

    summary_rows = [
        ("Root Type", root_report.get("root_type", "")),
        ("Health Status", root_report.get("health_status", "")),
        ("Root Health Index", f"{root_report.get('root_health_index', 0)}/100"),
        ("Branch Density", root_report.get("branch_density", "")),
        ("Growth Direction", root_report.get("growth_direction", "")),
        ("Age Estimate", root_report.get("age_estimate", "")),
        ("Biomass", root_report.get("biomass", "")),
        ("Soil Type", root_report.get("soil_type", "")),
        ("Soil Compaction", root_report.get("soil_compaction", ""))
    ]

    metrics_rows = [
        ("Symmetry Index", root_report.get("symmetry_index", 0.0)),
        ("Water Efficiency", f"{root_report.get('water_efficiency', 0)}%"),
        ("Nutrient Efficiency", f"{root_report.get('nutrient_efficiency', 0)}%"),
        ("Branch Points", root_report.get("branch_points", 0)),
        ("End Points", root_report.get("end_points", 0)),
        ("Branching Factor", root_report.get("branching_factor", 0.0)),
        ("Root Density", root_report.get("root_density", 0.0)),
        ("Root Length Index", root_report.get("root_length_index", 0.0)),
        ("Avg Thickness", root_report.get("avg_thickness", 0.0)),
        ("Thickness Variation", root_report.get("thickness_variation", 0.0)),
        ("Root Area", root_report.get("root_area", 0)),
        ("Avg Root Density", root_report.get("avg_root_density", 0.0)),
        ("Root System Depth", root_report.get("root_system_depth", 0)),
        ("Root System Width", root_report.get("root_system_width", 0)),
        ("Skeleton Depth", root_report.get("skeleton_depth", 0)),
        ("Skeleton Width", root_report.get("skeleton_width", 0)),
        ("Root Distribution X", root_report.get("root_distribution_x", 0.0)),
        ("Root Distribution Y", root_report.get("root_distribution_y", 0.0)),
        ("Root Tip Count", root_report.get("root_tip_count", 0)),
        ("Top Angle", root_report.get("top_angle", 0.0)),
        ("Bottom Angle", root_report.get("bottom_angle", 0.0)),
        ("Angle Mean", root_report.get("angle_mean", 0.0)),
        ("Angle Min", root_report.get("angle_min", 0.0)),
        ("Angle Max", root_report.get("angle_max", 0.0)),
        ("Adventitious Count", root_report.get("adventitious_count", 0)),
        ("Basal Count", root_report.get("basal_count", 0)),
        ("Adventitious Angle", root_report.get("adventitious_angle", 0.0)),
        ("Basal Angle", root_report.get("basal_angle", 0.0)),
        ("Taproot Diameter", root_report.get("taproot_diameter", 0.0)),
        ("Hypocotyl Diameter", root_report.get("hypocotyl_diameter", 0.0)),
        ("Crown Projection 25%", root_report.get("cp_dia25", 0)),
        ("Crown Projection 50%", root_report.get("cp_dia50", 0)),
        ("Crown Projection 75%", root_report.get("cp_dia75", 0)),
        ("Crown Projection 90%", root_report.get("cp_dia90", 0)),
        ("Nodal Length", root_report.get("nodal_length", 0.0)),
        ("Nodal Avg Diameter", root_report.get("nodal_avg_diameter", 0.0)),
        ("Lateral Branch Freq", root_report.get("lateral_branch_freq", 0.0)),
        ("Lateral Avg Length", root_report.get("lateral_avg_length", 0.0)),
        ("Lateral Angle Mean", root_report.get("lateral_angle_mean", 0.0)),
        ("Lateral Angle Min", root_report.get("lateral_angle_min", 0.0)),
        ("Lateral Angle Max", root_report.get("lateral_angle_max", 0.0))
    ]

    disease = root_report.get("disease_risk", {})
    disease_rows = [
        ("Root Rot Risk", disease.get("root_rot", "")),
        ("Fungal Risk", disease.get("fungal", "")),
        ("Damage Risk", disease.get("damage", ""))
    ]

    diameter_pcts = root_report.get("diameter_percentiles", {})
    skel_pcts = root_report.get("skeleton_diameter_percentiles", {})
    diameter_rows = [(f"D{p}", diameter_pcts.get(f"D{p}", 0.0)) for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]]
    skel_rows = [(f"DS{p}", skel_pcts.get(f"DS{p}", 0.0)) for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]]

    full_kv_rows = [(k, json.dumps(v, default=str)[:80] + "..." if isinstance(v, (dict, list)) and len(json.dumps(v, default=str)) > 80 else (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)) for k, v in sorted(root_report.items())]

    add_section("Summary", summary_rows)
    add_section("Metrics", metrics_rows)
    add_section("Disease Risk", disease_rows)
    add_section("Diameter Percentiles", diameter_rows)
    add_section("Skeleton Diameter Percentiles", skel_rows)
    add_section("Full Root Report (All Keys)", full_kv_rows)

    return pdf.output(dest="S").encode("latin-1")

# -------------------------------------------------
# HEADER
# -------------------------------------------------
# Enhanced header is rendered above; avoid duplicate header here.

# -------------------------------------------------
# PROGRESS STEPPER UI
# -------------------------------------------------
# -------------------------------------------------
# PROGRESS STEPPER UI
# -------------------------------------------------
# Using native Streamlit columns for better compatibility
step_cols = st.columns(4)
steps_data = [("📤", "Upload", 0), ("🔍", "Analysis", 1), ("🌱", "Simulate", 2), ("📊", "Report", 3)]
current_step = st.session_state.current_step

for i, (icon, label, step_num) in enumerate(steps_data):
    with step_cols[i]:
        if step_num < current_step:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <div style="width: 50px; height: 50px; border-radius: 50%; background: linear-gradient(135deg, #10b981, #059669); 
                            display: flex; align-items: center; justify-content: center; margin: 0 auto; 
                            box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);">
                    <span style="color: white; font-size: 1.25rem;">✓</span>
                </div>
                <div style="margin-top: 0.5rem; font-weight: 600; color: #10b981; font-size: 0.85rem;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
        elif step_num == current_step:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <div style="width: 50px; height: 50px; border-radius: 50%; background: linear-gradient(135deg, #0ea5e9, #0284c7); 
                            display: flex; align-items: center; justify-content: center; margin: 0 auto;
                            box-shadow: 0 4px 20px rgba(14, 165, 233, 0.4);">
                    <span style="font-size: 1.25rem;">{icon}</span>
                </div>
                <div style="margin-top: 0.5rem; font-weight: 600; color: #0ea5e9; font-size: 0.85rem;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <div style="width: 50px; height: 50px; border-radius: 50%; background: #f5f5f7; 
                            border: 2px solid #e5e7eb; display: flex; align-items: center; justify-content: center; margin: 0 auto;">
                    <span style="font-size: 1.25rem; opacity: 0.5;">{icon}</span>
                </div>
                <div style="margin-top: 0.5rem; font-weight: 600; color: #9ca3af; font-size: 0.85rem;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------
# INTERACTIVE DASHBOARD HOME
# -------------------------------------------------
with st.expander("📊 Dashboard Overview", expanded=True):
    # Stats Cards using native columns
    stats = st.session_state.dashboard_stats
    stat_cols = st.columns(4)
    
    with stat_cols[0]:
        st.markdown(f"""
        <div style="background: #000000; border-radius: 20px; padding: 1.5rem; text-align: center; 
                    border: 1px solid rgba(255,255,255,0.15); border-top: 4px solid #10b981;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🌿</div>
            <div style="font-size: 2rem; font-weight: 800; color: #ffffff;">{stats['total_plants']}</div>
            <div style="font-size: 0.85rem; color: #f5f5f7;">Plants Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_cols[1]:
        st.markdown(f"""
        <div style="background: #000000; border-radius: 20px; padding: 1.5rem; text-align: center; 
                    border: 1px solid rgba(255,255,255,0.15); border-top: 4px solid #8b5cf6;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧬</div>
            <div style="font-size: 2rem; font-weight: 800; color: #ffffff;">{stats['total_roots']}</div>
            <div style="font-size: 0.85rem; color: #f5f5f7;">Roots Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_cols[2]:
        st.markdown(f"""
        <div style="background: #000000; border-radius: 20px; padding: 1.5rem; text-align: center; 
                    border: 1px solid rgba(255,255,255,0.15); border-top: 4px solid #f59e0b;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">💚</div>
            <div style="font-size: 2rem; font-weight: 800; color: #ffffff;">{stats['avg_health_score']:.0f}</div>
            <div style="font-size: 0.85rem; color: #f5f5f7;">Avg Health Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_cols[3]:
        st.markdown(f"""
        <div style="background: #000000; border-radius: 20px; padding: 1.5rem; text-align: center; 
                    border: 1px solid rgba(255,255,255,0.15); border-top: 4px solid #06b6d4;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📈</div>
            <div style="font-size: 2rem; font-weight: 800; color: #ffffff;">{stats['total_simulations']}</div>
            <div style="font-size: 0.85rem; color: #f5f5f7;">Simulations Run</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two columns: Trend Chart and History
    dash_col1, dash_col2 = st.columns([1.5, 1])
    
    with dash_col1:
        st.markdown("""
        <div style="background: #000000; border-radius: 20px; padding: 1.5rem; 
                    border: 1px solid rgba(255,255,255,0.15);">
            <div style="font-size: 1.1rem; font-weight: 700; color: #ffffff; margin-bottom: 1rem;">📈 Analysis Trends</div>
        """, unsafe_allow_html=True)
        
        trend_chart = create_trend_chart(st.session_state.dashboard_stats)
        if trend_chart:
            st.plotly_chart(trend_chart, use_container_width=True, config={'displayModeBar': False})
        else:
            st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>📈</div>
                <div style='color: #ffffff; font-weight: 600; font-size: 1rem;'>Run analyses to see trends</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with dash_col2:
        history = st.session_state.analysis_history
        st.markdown("""
        <div style="background: #000000; border-radius: 20px; overflow: hidden; 
                    border: 1px solid rgba(255,255,255,0.15);">
            <div style="padding: 1rem 1.5rem; background: rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.1);">
                <span style="font-size: 1.1rem; font-weight: 700; color: #ffffff;">📋 Recent Analyses</span>
            </div>
        """, unsafe_allow_html=True)
        
        if not history:
            st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>📊</div>
                <div style='color: #ffffff; font-weight: 600; font-size: 1rem;'>No analyses yet</div>
                <div style='font-size: 0.9rem; color: #a0a0a0; margin-top: 0.25rem;'>Upload an image to get started</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for item in history[-5:][::-1]:
                badge_color = "#10b981" if item.get('health_score', 0) >= 70 else ("#f59e0b" if item.get('health_score', 0) >= 40 else "#ef4444")
                badge_text = "Healthy" if item.get('health_score', 0) >= 70 else ("Stressed" if item.get('health_score', 0) >= 40 else "Critical")
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-size: 1.5rem;">{item.get('icon', '🌱')}</div>
                    <div style="flex: 1;">
                        <div style="font-weight: 600; color: #ffffff; font-size: 0.9rem;">{item.get('name', 'Analysis')}</div>
                        <div style="font-size: 0.75rem; color: #888888;">
                            {item.get('timestamp', '')} 
                            <span style="background: {badge_color}22; color: {badge_color}; padding: 2px 6px; border-radius: 4px; font-weight: 600; margin-left: 0.5rem;">{badge_text}</span>
                        </div>
                    </div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff;">{item.get('health_score', 0)}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# LLM STATUS CHECK
# -------------------------------------------------
with st.expander("🔍 LLM Status", expanded=False):
    st.caption("Run a quick connectivity check for the LLM.")
    if st.button("Run LLM Status Check"):
        status = llm_health_check()
        if status.get("ok"):
            st.success(f"LLM reachable (model: {status.get('model', 'gpt-4o-mini')}).")
        else:
            st.error(f"LLM check failed: {status.get('error', 'Unknown error')}")

# -------------------------------------------------
# STICKY ACTION BAR + TIMELINE PANEL
# -------------------------------------------------
# -------------------------------------------------
# STICKY ACTION BAR + TIMELINE PANEL
# -------------------------------------------------
st.markdown(
        """
        <div class="sticky-action-bar">
            <div class="sticky-action-inner">
                <div style="font-weight: 600; font-size: 1.1rem;">SmartRoot AI</div>
                <div style="display: flex; gap: 1.5rem;">
                    <a class="sticky-action-btn" href="#upload-section">Upload</a>
                    <a class="sticky-action-btn" href="#root-analysis">Analysis</a>
                    <a class="sticky-action-btn" href="#simulate-root">Simulate</a>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
)

# -------------------------------------------------
# INPUT CONTROLS
# -------------------------------------------------
st.markdown("<a id='upload-section'></a>", unsafe_allow_html=True)

with st.expander("📋 Upload & Configure Analysis", expanded=True):
    st.markdown("""
    <p style='color: #a0a0a0; margin-bottom: 1.5rem;'>Securely upload your plant images and configure simulation parameters.</p>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns([1.5, 1])

    with c1:
        # Plant Image Section Header
        st.markdown("""
        <div style='background: #111111; border-radius: 12px; padding: 0.75rem 1rem; margin-bottom: 0.75rem; border-left: 3px solid #10b981; display: inline-block;'>
            <span style='font-size: 0.8rem; color: #10b981; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;'>🌿 Plant Analysis</span>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📸 Upload Vetiver Plant Image",
            type=["jpg", "png", "jpeg"],
            help="Select a clear image of your vetiver plant for analysis"
        )
        
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        
        # Root Image Section Header
        st.markdown("""
        <div style='background: #111111; border-radius: 12px; padding: 0.75rem 1rem; margin-bottom: 0.75rem; border-left: 3px solid #8b5cf6; display: inline-block;'>
            <span style='font-size: 0.8rem; color: #8b5cf6; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;'>🧬 Root Analysis</span>
        </div>
        """, unsafe_allow_html=True)
        root_image_file = st.file_uploader(
            "🪴 Upload Root Image",
            type=["jpg", "png", "jpeg"],
            help="Upload a clear root image for root intelligence analysis"
        )

    with c2:
        # Environment Settings Section Header
        st.markdown("""
        <div style='background: #111111; border-radius: 12px; padding: 0.75rem 1rem; margin-bottom: 0.75rem; border-left: 3px solid #0ea5e9; display: inline-block;'>
            <span style='font-size: 0.8rem; color: #0ea5e9; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;'>🌍 Environment Settings</span>
        </div>
        """, unsafe_allow_html=True)
        soil_type = st.selectbox(
            "🌍 Soil Type",
            ["Sandy", "Clay", "Loamy"],
            help="Select the type of soil where the plant is grown"
        )
        
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        
        # Performance Options Section Header
        st.markdown("""
        <div style='background: #111111; border-radius: 12px; padding: 0.75rem 1rem; margin-bottom: 0.75rem; border-left: 3px solid #f59e0b; display: inline-block;'>
            <span style='font-size: 0.8rem; color: #f59e0b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;'>⚡ Performance Options</span>
        </div>
        """, unsafe_allow_html=True)
        fast_overall = st.checkbox(
            "⚡ Fast overall analysis (skip LLM image species)",
            value=True,
            help="Speeds up by skipping LLM-based species classification"
        )
        fast_root_analysis = st.checkbox(
            "⚡ Fast root analysis (less detail)",
            value=True,
            help="Speeds up root analysis by reducing resolution and skipping heavy metrics"
        )

plant_image_bytes = uploaded_file.getvalue() if uploaded_file else None
root_image_bytes = root_image_file.getvalue() if root_image_file else None

# -------------------------------------------------
# ROOT IMAGE INTELLIGENCE (OPTIONAL)
# -------------------------------------------------
if root_image_file:

    try:
        skeleton_placeholder = st.empty()
        skeleton_placeholder.markdown(
            """
            <div class="section-card">
                <div class="skeleton skeleton-line" style="width:40%"></div>
                <div class="skeleton skeleton-line" style="width:70%"></div>
                <div class="skeleton skeleton-block"></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        with st.spinner("Analyzing root image..."):
            if fast_root_analysis:
                root_report = cached_analyze_root_image_fast(root_image_bytes)
            else:
                root_report = cached_analyze_root_image(root_image_bytes)
            if fast_overall:
                root_species = cached_classify_root_species_fast(root_image_bytes)
            else:
                root_species = cached_classify_root_species(root_image_bytes)
        skeleton_placeholder.empty()
        
        # Update progress step and dashboard stats
        st.session_state.current_step = 1
        root_health = root_report.get('root_health_index', 50)
        root_moisture = root_report.get('water_efficiency', 50)
        root_nutrient = root_report.get('nutrient_efficiency', 50)
        update_dashboard_stats('root', health_score=root_health, moisture=root_moisture, nutrient=root_nutrient)
        add_to_analysis_history(f"Root - {root_species.get('species', 'Unknown')[:15]}", "🧬", root_health, 'root', root_moisture, root_nutrient)
        
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<a id='root-analysis'></a>", unsafe_allow_html=True)
        st.markdown("<div class='section-separator'>🧬 Root Image Intelligence</div>", unsafe_allow_html=True)

        # Circular Health Score Visualization
        health_col1, health_col2, health_col3 = st.columns([1, 1.2, 1])
        with health_col2:
            st.markdown(render_circular_health_score(root_health, "Root Health"), unsafe_allow_html=True)
        
        # Mini gauges for key metrics in a horizontal row
        symmetry_pct = min(100, int(root_report.get('symmetry_index', 0) * 100))
        st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: flex-start; gap: 3rem; padding: 1.5rem 0; flex-wrap: wrap;">
            {render_mini_gauge(root_report.get('water_efficiency', 0), "Water", "#0ea5e9")}
            {render_mini_gauge(root_report.get('nutrient_efficiency', 0), "Nutrient", "#10b981")}
            {render_mini_gauge(symmetry_pct, "Symmetry", "#8b5cf6")}
        </div>
        """, unsafe_allow_html=True)

        # Root Intelligence Summary Box
        st.markdown(f"""
<div class='stress-result-box'>
    <div class='stress-label'>🧬 Analysis Result</div>
    <div class='stress-value'>{root_report.get('root_type', 'Unknown')}</div>
    <div style='display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;'>
        <div class='stress-reliability-pill'>
            <span>🩺</span>
            <span>{root_report.get('health_status', 'Unknown')}</span>
        </div>
        <div class='stress-reliability-pill'>
            <span>🌿</span>
            <span>Species: {root_species.get('species', 'Unknown')}</span>
        </div>
        <div class='stress-reliability-pill'>
             <span>📊</span>
             <span>Health Index: {root_report.get('root_health_index', 0)}/100</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        # LLM-Powered Root Analysis Explanation
        with st.expander("🤖 AI Root Analysis Insights", expanded=True):
            with st.spinner("Generating AI insights..."):
                root_explanation = llm_explain_root_analysis(
                    species=root_species.get('species', 'Unknown'),
                    health_index=root_report.get('root_health_index', 50),
                    water_efficiency=root_report.get('water_efficiency', 50),
                    nutrient_efficiency=root_report.get('nutrient_efficiency', 50),
                    root_type=root_report.get('root_type', 'Unknown')
                )
            st.markdown(f"""
            <div style="background: #000000; border-radius: 16px; padding: 1.5rem; border: 1px solid rgba(255,255,255,0.15);">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                    <span style="font-size: 1.5rem;">🧬</span>
                    <span style="font-size: 1.1rem; font-weight: 700; color: #10b981;">Root Type: {root_report.get('root_type', 'Unknown')}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                    <span style="font-size: 1.5rem;">🌿</span>
                    <span style="font-size: 1.1rem; font-weight: 700; color: #8b5cf6;">Species: {root_species.get('species', 'Unknown')}</span>
                </div>
                <div style="color: #ffffff; line-height: 1.7; font-size: 0.95rem;">
                    {root_explanation.replace(chr(10), '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Detailed Metrics Grid
        st.markdown(f"""
<div class='analytics-grid'>
    <div class='analytics-box'>
        <div class='analytics-icon'>📐</div>
        <div class='analytics-value'>{root_report.get('symmetry_index', 0.0):.2f}</div>
        <div class='analytics-label'>Symmetry Index</div>
    </div>
    <div class='analytics-box'>
        <div class='analytics-icon'>💧</div>
        <div class='analytics-value'>{root_report.get('water_efficiency', 0)}%</div>
        <div class='analytics-label'>Water Absorption</div>
    </div>
    <div class='analytics-box'>
        <div class='analytics-icon'>🌱</div>
        <div class='analytics-value'>{root_report.get('nutrient_efficiency', 0)}%</div>
        <div class='analytics-label'>Nutrient Uptake</div>
    </div>
    <div class='analytics-box'>
        <div class='analytics-icon'>🔍</div>
        <div class='analytics-value'>{root_species.get('confidence', 0.0):.0%}</div>
        <div class='analytics-label'>ID Confidence</div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.table([
            {"Trait": "Branch Density", "Value": str(root_report.get("branch_density", ""))},
            {"Trait": "Growth Direction", "Value": str(root_report.get("growth_direction", ""))},
            {"Trait": "Root Age", "Value": str(root_report.get("age_estimate", ""))},
            {"Trait": "Biomass", "Value": str(root_report.get("biomass", ""))},
            {"Trait": "Soil Type", "Value": str(root_report.get("soil_type", ""))},
            {"Trait": "Soil Compaction", "Value": str(root_report.get("soil_compaction", ""))},
            {"Trait": "Branch Points", "Value": str(root_report.get("branch_points", 0))},
            {"Trait": "End Points", "Value": str(root_report.get("end_points", 0))},
            {"Trait": "Branching Factor", "Value": str(root_report.get("branching_factor", 0.0))},
            {"Trait": "Root Density", "Value": str(root_report.get("root_density", 0.0))},
            {"Trait": "Root Length Index", "Value": str(root_report.get("root_length_index", 0.0))},
            {"Trait": "Avg Thickness", "Value": str(root_report.get("avg_thickness", 0.0))},
            {"Trait": "Thickness Variation", "Value": str(root_report.get("thickness_variation", 0.0))}
        ])

        with st.expander("📊 Detailed Root Geometry (Depth/Width/Angles)"):
            st.table([
                {"Trait": "Root Area", "Value": str(root_report.get("root_area", 0))},
                {"Trait": "Avg Root Density", "Value": str(root_report.get("avg_root_density", 0.0))},
                {"Trait": "Root System Depth", "Value": str(root_report.get("root_system_depth", 0))},
                {"Trait": "Root System Width", "Value": str(root_report.get("root_system_width", 0))},
                {"Trait": "Skeleton Depth", "Value": str(root_report.get("skeleton_depth", 0))},
                {"Trait": "Skeleton Width", "Value": str(root_report.get("skeleton_width", 0))},
                {"Trait": "Root Distribution X", "Value": str(root_report.get("root_distribution_x", 0.0))},
                {"Trait": "Root Distribution Y", "Value": str(root_report.get("root_distribution_y", 0.0))},
                {"Trait": "Root Tip Count", "Value": str(root_report.get("root_tip_count", 0))},
                {"Trait": "Top Angle", "Value": str(root_report.get("top_angle", 0.0))},
                {"Trait": "Bottom Angle", "Value": str(root_report.get("bottom_angle", 0.0))},
                {"Trait": "Angle Mean", "Value": str(root_report.get("angle_mean", 0.0))},
                {"Trait": "Angle Min", "Value": str(root_report.get("angle_min", 0.0))},
                {"Trait": "Angle Max", "Value": str(root_report.get("angle_max", 0.0))}
            ])

        diameter_pcts = root_report.get("diameter_percentiles", {})
        skel_pcts = root_report.get("skeleton_diameter_percentiles", {})
        with st.expander("📐 Diameter Percentiles (D10–D90)"):
            st.table([
                {"Trait": "D10", "Value": str(diameter_pcts.get("D10", 0.0))},
                {"Trait": "D20", "Value": str(diameter_pcts.get("D20", 0.0))},
                {"Trait": "D30", "Value": str(diameter_pcts.get("D30", 0.0))},
                {"Trait": "D40", "Value": str(diameter_pcts.get("D40", 0.0))},
                {"Trait": "D50", "Value": str(diameter_pcts.get("D50", 0.0))},
                {"Trait": "D60", "Value": str(diameter_pcts.get("D60", 0.0))},
                {"Trait": "D70", "Value": str(diameter_pcts.get("D70", 0.0))},
                {"Trait": "D80", "Value": str(diameter_pcts.get("D80", 0.0))},
                {"Trait": "D90", "Value": str(diameter_pcts.get("D90", 0.0))}
            ])

            st.table([
                {"Trait": "DS10", "Value": str(skel_pcts.get("DS10", 0.0))},
                {"Trait": "DS20", "Value": str(skel_pcts.get("DS20", 0.0))},
                {"Trait": "DS30", "Value": str(skel_pcts.get("DS30", 0.0))},
                {"Trait": "DS40", "Value": str(skel_pcts.get("DS40", 0.0))},
                {"Trait": "DS50", "Value": str(skel_pcts.get("DS50", 0.0))},
                {"Trait": "DS60", "Value": str(skel_pcts.get("DS60", 0.0))},
                {"Trait": "DS70", "Value": str(skel_pcts.get("DS70", 0.0))},
                {"Trait": "DS80", "Value": str(skel_pcts.get("DS80", 0.0))},
                {"Trait": "DS90", "Value": str(skel_pcts.get("DS90", 0.0))}
            ])

        with st.expander("🌱 Structural Counts & Diameters"):
            st.table([
                {"Trait": "Adventitious Count", "Value": str(root_report.get("adventitious_count", 0))},
                {"Trait": "Basal Count", "Value": str(root_report.get("basal_count", 0))},
                {"Trait": "Adventitious Angle", "Value": str(root_report.get("adventitious_angle", 0.0))},
                {"Trait": "Basal Angle", "Value": str(root_report.get("basal_angle", 0.0))},
                {"Trait": "Taproot Diameter", "Value": str(root_report.get("taproot_diameter", 0.0))},
                {"Trait": "Hypocotyl Diameter", "Value": str(root_report.get("hypocotyl_diameter", 0.0))},
                {"Trait": "Crown Projection 25%", "Value": str(root_report.get("cp_dia25", 0))},
                {"Trait": "Crown Projection 50%", "Value": str(root_report.get("cp_dia50", 0))},
                {"Trait": "Crown Projection 75%", "Value": str(root_report.get("cp_dia75", 0))},
                {"Trait": "Crown Projection 90%", "Value": str(root_report.get("cp_dia90", 0))}
            ])

        with st.expander("🧷 Nodal & Lateral Branching"):
            st.table([
                {"Trait": "Nodal Length", "Value": str(root_report.get("nodal_length", 0.0))},
                {"Trait": "Nodal Avg Diameter", "Value": str(root_report.get("nodal_avg_diameter", 0.0))},
                {"Trait": "Lateral Branch Freq", "Value": str(root_report.get("lateral_branch_freq", 0.0))},
                {"Trait": "Lateral Avg Length", "Value": str(root_report.get("lateral_avg_length", 0.0))},
                {"Trait": "Lateral Angle Mean", "Value": str(root_report.get("lateral_angle_mean", 0.0))},
                {"Trait": "Lateral Angle Min", "Value": str(root_report.get("lateral_angle_min", 0.0))},
                {"Trait": "Lateral Angle Max", "Value": str(root_report.get("lateral_angle_max", 0.0))}
            ])

        try:
            report_pdf = _build_root_image_report_pdf(
                root_report,
                root_species,
                root_image_bytes,
                filename_hint=root_image_file.name if root_image_file else "root_image.png"
            )

            st.download_button(
                "⬇️ Download Root Image Report (PDF)",
                data=report_pdf,
                file_name="root_image_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"Report generation failed: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Root image analysis failed: {e}")

# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
if uploaded_file:
    # Update progress step
    st.session_state.current_step = 1
    
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<a id='plant-analysis'></a>", unsafe_allow_html=True)
    st.markdown("<div class='section-separator'>🌿 Plant Stress Analysis</div>", unsafe_allow_html=True)
    with st.spinner("Analyzing plant image..."):
        if fast_overall:
            species_result = cached_classify_plant_species_fast(plant_image_bytes)
        else:
            species_result = cached_classify_plant_species(plant_image_bytes)
    st.subheader("🌿 Plant Species Identification")
    sp1, sp2 = st.columns(2)
    with sp1:
        st.metric("Species", species_result.get("species", ""))
    with sp2:
        st.metric("Confidence", f"{species_result.get('confidence', 0.0):.0%}")

    if species_result.get("species", "").strip().lower() not in {"vetiver"}:
        st.markdown("""
        <div style="background: #000000; border: 1px solid #f59e0b; border-radius: 12px; padding: 1rem 1.5rem; margin: 1rem 0;">
            <span style="color: #f59e0b; font-weight: 600;">⚠️ Detected species is not Vetiver. Continuing with stress analysis.</span>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------
    # CNN + LLM DOMINANCE LOGIC
    # -------------------------------------------------
    with st.spinner("Running stress analysis..."):
        label, moisture, nutrient, confidence, cnn_weak = cached_predict_stress(plant_image_bytes)
    
    # Calculate a health score based on moisture, nutrient, and confidence
    plant_health_score = int((moisture + nutrient + (confidence * 100)) / 3)
    
    # Update dashboard stats
    update_dashboard_stats('plant', health_score=plant_health_score, moisture=moisture, nutrient=nutrient)
    add_to_analysis_history(f"Plant - {species_result.get('species', 'Unknown')[:12]}", "🌿", plant_health_score, 'plant', moisture, nutrient)

    st.markdown("<div class='results-section-card'>", unsafe_allow_html=True)
    st.markdown("### 🧠 Stress Assessment Results")
    
    # Circular Health Score Visualization for Plant
    health_col1, health_col2, health_col3 = st.columns([1, 1.2, 1])
    with health_col2:
        st.markdown(render_circular_health_score(plant_health_score, "Plant Health"), unsafe_allow_html=True)
    
    # Mini gauges for moisture and nutrients in a horizontal row
    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: flex-start; gap: 3rem; padding: 1.5rem 0; flex-wrap: wrap;">
        {render_mini_gauge(moisture, "Moisture", "#0ea5e9")}
        {render_mini_gauge(nutrient, "Nutrients", "#10b981")}
        {render_mini_gauge(int(confidence * 100), "Confidence", "#f59e0b")}
    </div>
    """, unsafe_allow_html=True)

    if cnn_weak:
        st.markdown("""
        <div class='stAlert' style='background-color: #000000; border: 1px solid #f59e0b; border-radius: 12px; padding: 1rem;'>
            <div style='display: flex; gap: 1rem; align-items: start;'>
                <div style='font-size: 1.5rem;'>⚠️</div>
                <div>
                    <div style='font-weight: 600; margin-bottom: 0.25rem; color: #f59e0b;'>CNN Confidence is Low</div>
                    <div style='font-size: 0.95rem; color: #ffffff;'>AI biological reasoning dominates this analysis</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Enhanced metrics dashboard with horizontal layout
    st.markdown("""
    <div class='metrics-dashboard'>
        <div class='metric-card moisture'>
            <div class='metric-label'>💧 Moisture Level</div>
            <div class='metric-value'>""" + f"{moisture}%" + """</div>
            <div class='metric-subtitle'>Soil hydration</div>
        </div>
        <div class='metric-card nutrients'>
            <div class='metric-label'>🌿 Nutrients Level</div>
            <div class='metric-value'>""" + f"{nutrient}%" + """</div>
            <div class='metric-subtitle'>Plant nutrition</div>
        </div>
        <div class='metric-card confidence'>
            <div class='metric-label'>🔬 CNN Confidence</div>
            <div class='metric-value'>""" + f"{confidence:.0%}" + """</div>
            <div class='metric-subtitle'>Model certainty</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Combined Stress Result Box
    status_icon = "🟢" if not cnn_weak else "🟠"
    reliability_text = "High Confidence" if not cnn_weak else "Low Confidence"
    reliability_color = "#34c759" if not cnn_weak else "#ff9500"
    
    st.markdown(f"""
<div class='stress-result-box'>
    <div class='stress-label'>🎯 Detected Stress Type</div>
    <div class='stress-value'>{label}</div>
    <div style='display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;'>
        <div class='stress-reliability-pill'>
            <span>{status_icon}</span>
            <span style='color: {reliability_color};'>{reliability_text}</span>
        </div>
        <div class='stress-reliability-pill'>
            <span>🔬</span>
            <span>CNN Confidence: {confidence:.0%}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # LLM-Powered Plant Analysis Explanation
    with st.expander("🤖 AI Plant Analysis Insights", expanded=True):
        with st.spinner("Generating AI insights..."):
            plant_explanation = llm_explain_plant_analysis(
                species=species_result.get('species', 'Unknown'),
                health_score=plant_health_score,
                moisture=moisture,
                nutrient=nutrient,
                stress_label=label
            )
        st.markdown(f"""
        <div style="background: #000000; border-radius: 16px; padding: 1.5rem; border: 1px solid rgba(255,255,255,0.15);">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">🌿</span>
                <span style="font-size: 1.1rem; font-weight: 700; color: #10b981;">Plant Type: {species_result.get('species', 'Unknown')}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">🎯</span>
                <span style="font-size: 1.1rem; font-weight: 700; color: #f59e0b;">Stress Status: {label}</span>
            </div>
            <div style="color: #ffffff; line-height: 1.7; font-size: 0.95rem;">
                {plant_explanation.replace(chr(10), '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # -------------------------------------------------
    # PARAMETERS USED (USER-SELECTED / FINAL VALUES)
    # -------------------------------------------------
    st.markdown("<div class='parameters-section-card'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Analysis Parameters & Values")

    # Parameter explanation with tooltips
    st.markdown("""
    <div style='background: rgba(240, 253, 244, 0.5); border-radius: 0.75rem; padding: 1rem; margin-bottom: 1.5rem;'>
        <p style='color: #475569; font-size: 0.95rem; margin: 0;'>
            <span style='font-weight: 600;'>These parameters</span> are derived from your selected soil type and the CNN analysis of your plant image. They influence the root growth simulation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    u1, u2, u3 = st.columns(3)

    with u1:
        st.markdown(f"""
        <div style='background: rgba(255, 255, 255, 0.6); border-left: 4px solid #10b981; border-radius: 0.75rem; padding: 1.5rem; text-align: center;'>
            <div style='font-size: 0.85rem; font-weight: 600; color: #475569; text-transform: uppercase; margin-bottom: 0.5rem;'>🌍 Soil Type</div>
            <div style='font-size: 1.5rem; font-weight: 800; color: #10b981;'>{soil_type}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with u2:
        moisture_visual = "💧" if moisture > 50 else "🏜️"
        st.markdown(f"""
        <div style='background: rgba(255, 255, 255, 0.6); border-left: 4px solid #0ea5e9; border-radius: 0.75rem; padding: 1.5rem; text-align: center;'>
            <div style='font-size: 0.85rem; font-weight: 600; color: #475569; text-transform: uppercase; margin-bottom: 0.5rem;'>{moisture_visual} Moisture</div>
            <div style='font-size: 1.5rem; font-weight: 800; color: #0ea5e9;'>{moisture}%</div>
            <div class='gauge-container' style='margin-top: 0.75rem;'>
                <div class='gauge-bar'>
                    <div class='gauge-fill moisture' style='width: {moisture}%;'>
                        <span class='gauge-percentage' style='display: {("inline" if moisture > 10 else "none")};'>{moisture}%</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with u3:
        nutrient_visual = "🌱" if nutrient > 50 else "⚠️"
        st.markdown(f"""
        <div style='background: rgba(255, 255, 255, 0.6); border-left: 4px solid #22c55e; border-radius: 0.75rem; padding: 1.5rem; text-align: center;'>
            <div style='font-size: 0.85rem; font-weight: 600; color: #475569; text-transform: uppercase; margin-bottom: 0.5rem;'>{nutrient_visual} Nutrients</div>
            <div style='font-size: 1.5rem; font-weight: 800; color: #22c55e;'>{nutrient}%</div>
            <div class='gauge-container' style='margin-top: 0.75rem;'>
                <div class='gauge-bar'>
                    <div class='gauge-fill nutrients' style='width: {nutrient}%;'>
                        <span class='gauge-percentage' style='display: {("inline" if nutrient > 10 else "none")};'>{nutrient}%</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------
    # RECOMMENDED ACTIONS BASED ON ANALYSIS
    # -------------------------------------------------
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("### 💡 Recommendations")

    actions = []
    if moisture < 45:
        actions.append(("💧", "Increase irrigation", "Soil moisture is low. Provide consistent watering to avoid stress."))
    elif moisture > 70:
        actions.append(("🌧️", "Improve drainage", "Soil moisture is high. Ensure proper drainage to prevent root rot."))
    else:
        actions.append(("✅", "Moisture optimal", f"Soil moisture at {moisture}% is within a healthy range."))

    if nutrient < 45:
        actions.append(("🌿", "Apply fertilizer", "Nutrient level is low. Add balanced fertilizer to improve growth."))
    else:
        actions.append(("🧪", "Maintain nutrients", "Nutrient availability is adequate. Maintain current regimen."))

    if label and isinstance(label, str):
        actions.append(("🩺", "Monitor stress", f"Detected stress state: {label}. Monitor plant response and adjust care."))
    
    for i, (icon, title, description) in enumerate(actions, 1):
        st.markdown(f"""
        <div style='display: flex; gap: 1rem; padding: 1rem 0; border-bottom: 1px solid rgba(0,0,0,0.05);'>
            <div style='background: #f5f5f7; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.25rem;'>{icon}</div>
            <div>
                <div style='font-weight: 600; color: #1d1d1f; margin-bottom: 0.25rem;'>{title}</div>
                <div style='color: #86868b; font-size: 0.95rem; line-height: 1.4;'>{description}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------
    # SIMULATION TRIGGER WITH CTA
    # -------------------------------------------------
    st.markdown("<div class='section-separator'>🌱 Root Growth Simulation</div>", unsafe_allow_html=True)
    st.markdown("<a id='simulate-root'></a>", unsafe_allow_html=True)
    st.markdown("""
    <div style='margin: 2rem 0;'>
        <p style='text-align: center; color: #ffffff; font-size: 0.95rem; margin-bottom: 1rem;'>
            <strong>Ready to simulate root growth?</strong> Click below to generate a detailed visualization based on the analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🌿 Simulate Root Growth", use_container_width=True, type="primary"):
        # Update progress step
        st.session_state.current_step = 2
        
        with st.spinner("Simulating root growth..."):
            segments = cached_simulate_root(moisture, nutrient, soil_type)

            realism_score, ai_feedback = evaluate_root_realism(segments)
            render_cfg = ai_render_params(realism_score)

        # -------------------------------------------------
        # ANALYTICS
        # -------------------------------------------------
            max_depth = max(r["y2"] for r in segments)
            total_length = sum(
                math.dist((r["x1"], r["y1"]), (r["x2"], r["y2"]))
                for r in segments
            )
            spread = max(abs(r["x2"]) for r in segments)
            hair_count = sum(1 for r in segments if r["thickness"] < 0.7)

        # -------------------------------------------------
        # LLM BIOLOGICAL INTERPRETATION
        # -------------------------------------------------
            metrics_for_llm = {
                "max_depth_cm": round(max_depth, 2),
                "total_length_cm": round(total_length, 2),
                "horizontal_spread_cm": round(spread, 2),
                "root_hair_count": hair_count,
                "realism_score": realism_score,
                "cnn_confidence": round(confidence, 3),
                "cnn_reliability": "LOW" if cnn_weak else "HIGH"
            }

            llm_feedback = cached_llm_analysis(metrics_for_llm, soil_type)
        
        # Update dashboard stats for simulation
        update_dashboard_stats('simulation')
        
        # Update progress to report stage
        st.session_state.current_step = 3

        # -------------------------------------------------
        # VISUALIZATION TABS: 2D (Matplotlib) and 3D (Plotly)
        # -------------------------------------------------
        st.subheader("🌱 AI-Enhanced Root System")
        
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 2D View", "🌐 3D Interactive", "📈 Metrics Radar"])
        
        with viz_tab1:
            # Original matplotlib visualization
            fig, ax = plt.subplots(figsize=(3.3, 5.0), dpi=160)

            ax.axhline(0, color="#3b2f1e", linewidth=2)

            xs = [r["x1"] for r in segments] + [r["x2"] for r in segments]
            center_x = np.mean(xs)

            for r in segments:
                depth_ratio = min(r["y2"] / max_depth, 1.0)

                lw = r["thickness"] * render_cfg["thickness_scale"]
                lw *= (1 - 0.6 * depth_ratio)
                lw = max(lw, 0.6)

                alpha = render_cfg["alpha"] * (1 - depth_ratio * render_cfg["depth_fade"])
                alpha = min(max(alpha, 0.08), 1.0)

                color = (
                    0.55 - 0.4 * depth_ratio,
                    0.38 - 0.28 * depth_ratio,
                    0.18 - 0.15 * depth_ratio
                )

                ax.plot(
                    [(r["x1"] - center_x), (r["x2"] - center_x)],
                    [r["y1"], r["y2"]],
                    linewidth=lw,
                    color=color,
                    alpha=alpha,
                    solid_capstyle="round"
                )

            ax.set_xlim(min(xs) - center_x - 0.6, max(xs) - center_x + 0.6)
            ax.set_ylim(-0.3, max_depth + 0.8)
            ax.invert_yaxis()
            ax.axis("off")
            ax.set_facecolor("#1b140f")

            # Add parameters text on canvas
            params_text = f"Moisture: {moisture:.1f}% | Nutrient: {nutrient:.1f}% | Soil: {soil_type}"
            ax.text(
                0.5, 0.02, params_text,
                transform=ax.transAxes,
                ha='center', va='bottom',
                fontsize=10, color='#10b981',
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#2a2a2a', edgecolor='#10b981', linewidth=1.5)
            )

            st.pyplot(fig, use_container_width=False)
        
        with viz_tab2:
            # Interactive 3D Plotly visualization
            st.markdown("""
            <p style='color: #64748b; font-size: 0.9rem; text-align: center; margin-bottom: 1rem;'>
                🖱️ Drag to rotate • Scroll to zoom • Double-click to reset
            </p>
            """, unsafe_allow_html=True)
            
            fig_3d = create_root_3d_visualization(segments, max_depth)
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True, config={'displayModeBar': True})
        
        with viz_tab3:
            # Metrics radar chart
            st.markdown("""
            <p style='color: #64748b; font-size: 0.9rem; text-align: center; margin-bottom: 1rem;'>
                Root system metrics visualization
            </p>
            """, unsafe_allow_html=True)
            
            radar_fig = create_metrics_radar_chart(metrics_for_llm)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True, config={'displayModeBar': False})

        # -------------------------------------------------
        # DOWNLOAD
        # -------------------------------------------------
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        d1, d2, d3 = st.columns([1, 2, 1])
        with d2:
            st.download_button(
                "⬇️ Download Root Image (PNG)",
                data=buf,
                file_name="vetiver_root_ai.png",
                mime="image/png",
                use_container_width=True
            )

        st.divider()

        # -------------------------------------------------
        # ANALYTICS DASHBOARD - ENHANCED
        # -------------------------------------------------
        st.markdown("<div class='results-section-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Root System Analytics")
        
        st.markdown("""
        <p style='color: #475569; font-size: 0.95rem; margin: 0 0 1.5rem 0;'>
            Detailed metrics of the simulated root architecture based on your plant's stress state.
        </p>
        """, unsafe_allow_html=True)
        
        # Metrics in enhanced card layout
        st.markdown("""
<div class='analytics-grid'>
    <div class='analytics-box'>
        <div class='analytics-icon'>📏</div>
        <div class='analytics-value'>""" + f"{round(max_depth, 2)}" + """</div>
        <div class='analytics-label'>Max Depth (cm)</div>
    </div>
    <div class='analytics-box'>
        <div class='analytics-icon'>➰</div>
        <div class='analytics-value'>""" + f"{round(total_length, 2)}" + """</div>
        <div class='analytics-label'>Total Length (cm)</div>
    </div>
    <div class='analytics-box'>
        <div class='analytics-icon'>↔️</div>
        <div class='analytics-value'>""" + f"{round(spread, 2)}" + """</div>
        <div class='analytics-label'>Horiz. Spread (cm)</div>
    </div>
    <div class='analytics-box'>
        <div class='analytics-icon'>🔬</div>
        <div class='analytics-value'>""" + f"{hair_count}" + """</div>
        <div class='analytics-label'>Root Hairs</div>
    </div>
</div>
""", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------------------------
        # AI FEEDBACK & REALISM ASSESSMENT
        # -------------------------------------------------
        st.markdown("<div class='results-section-card'>", unsafe_allow_html=True)
        st.markdown("### 🤖 AI Realism Assessment")
        
        st.markdown(f"""
<div style='display: flex; align-items: center; gap: 2rem; margin-bottom: 1.5rem;'>
    <div style='flex: 0 0 auto;'>
        <div style='text-align: center; background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%); border-radius: 1rem; padding: 1.5rem;'>
            <div style='font-size: 0.85rem; font-weight: 600; color: #475569; text-transform: uppercase; margin-bottom: 0.5rem;'>Realism Score</div>
            <div style='font-size: 2rem; font-weight: 900; color: #10b981;'>{realism_score}</div>
            <div style='font-size: 0.9rem; color: #475569;'>/ 100</div>
        </div>
    </div>
    <div style='flex: 1;'>
        <h4 style='color: #0f172a; margin: 0 0 0.75rem 0; font-weight: 700;'>📋 Key Observations</h4>
        <ul style='margin: 0; padding-left: 1.5rem;'>
""", unsafe_allow_html=True)
        
        for observation in ai_feedback:
            st.markdown(f"<li style='color: #475569; margin-bottom: 0.5rem; line-height: 1.5;'>{observation}</li>", unsafe_allow_html=True)
        
        st.markdown("""
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style='height: 2px; background: linear-gradient(90deg, transparent, var(--primary-green), transparent); margin: 2rem 0;'></div>
        """, unsafe_allow_html=True)

        # -------------------------------------------------
        # AI BIOLOGICAL EXPLANATION - ENHANCED
        # -------------------------------------------------
        st.markdown("<div class='results-section-card'>", unsafe_allow_html=True)
        st.markdown("### 🧬 AI Biological Explanation")
        st.markdown("""
        <p style='color: #475569; font-size: 0.95rem; margin: 0 0 1.5rem 0;'>
            Based on the CNN analysis and root simulation, here's the biological interpretation of your plant's stress state and root development:
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='background: rgba(240, 253, 244, 0.5); border-left: 4px solid var(--primary-green); border-radius: 0.75rem; padding: 1.5rem; line-height: 1.8;'>
            {llm_feedback}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# 👇 THIS ELSE MUST ALIGN WITH: if uploaded_file:
else:
    # Reset progress step when no file is uploaded
    st.session_state.current_step = 0
    
    st.markdown("""
    <div style='text-align: center; padding: 6rem 1rem; max-width: 700px; margin: 0 auto;'>
        <div style='margin-bottom: 2rem; font-size: 4rem;'>🌱</div>
        <h2 style='color: #1d1d1f; font-size: 2.5rem; letter-spacing: -0.02em; margin-bottom: 1rem; border: none !important;'>Welcome to SmartRoot AI</h2>
        <p style='color: #86868b; font-size: 1.25rem; line-height: 1.5; margin-bottom: 3rem;'>
            Upload a vetiver plant image to analyze plant health and simulate root growth patterns with advanced AI.
        </p>
        <div style='display: inline-flex; gap: 1.5rem; margin-top: 1rem;'>
             <div style='text-align:left; background: #000000; padding: 2rem; border-radius: 20px; width: 280px; border: 1px solid rgba(255,255,255,0.15); box-shadow: 0 10px 30px rgba(0,0,0,0.2); transition: all 0.3s ease;'>
                <div style='font-size: 2rem; margin-bottom: 0.75rem;'>📷</div>
                <div style='font-weight: 700; margin-bottom: 0.5rem; color: #ffffff; font-size: 1.1rem;'>Upload Image</div>
                <div style='font-size: 0.9rem; color: #a0a0a0;'>Clear photo of your plant</div>
             </div>
             <div style='text-align:left; background: #000000; padding: 2rem; border-radius: 20px; width: 280px; border: 1px solid rgba(255,255,255,0.15); box-shadow: 0 10px 30px rgba(0,0,0,0.2); transition: all 0.3s ease;'>
                <div style='font-size: 2rem; margin-bottom: 0.75rem;'>🧠</div>
                <div style='font-weight: 700; margin-bottom: 0.5rem; color: #ffffff; font-size: 1.1rem;'>Get Analysis</div>
                <div style='font-size: 0.9rem; color: #a0a0a0;'>Instant health insights</div>
             </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

