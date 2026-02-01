import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def render_prediction_banner(label, confidence, target_species="Vetiver"):
    """
    Renders a color-coded prediction banner based on label and confidence.
    """
    label_lower = str(label).lower()
    
    # Logic for color and status icon
    if "healthy" in label_lower:
        status = "success"
        color = "#10b981"
        icon = "✅"
    elif any(x in label_lower for x in ["moderate", "stressed", "early"]):
        status = "warning"
        color = "#f59e0b"
        icon = "⚠️"
    else:
        status = "error"
        color = "#ef4444"
        icon = "🚨"

    conf_pct = int(confidence * 100)
    
    # Main Banner
    st.markdown(f"""
    <div style="background: {color}15; border-left: 5px solid {color}; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <span style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: {color}; font-weight: 700;">Prediction Result</span>
                <h2 style="margin: 0; color: white; display: flex; align-items: center; gap: 0.5rem;">
                    <span>{icon}</span> {label}
                </h2>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.9rem; color: #86868b;">Confidence</span>
                <div style="font-size: 2.5rem; font-weight: 800; color: {color}; line-height: 1;">{conf_pct}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Low Confidence Warning
    if confidence < 0.6:
        st.error(f"⚠️ **Low Confidence ({conf_pct}%)** — The AI is uncertain. Please ensure the plant is well-lit and centered in the frame, then retake the image.")

def render_root_geometry_plotly(report):
    """
    Renders root geometry metrics using Plotly Radar/Bar charts.
    """
    if not report:
        return

    # Diameter Percentiles
    diameter_pcts = report.get("diameter_percentiles", {})
    if diameter_pcts:
        labels = list(diameter_pcts.keys())
        values = list(diameter_pcts.values())
        
        fig = go.Figure(data=[go.Bar(
            x=labels, 
            y=values,
            marker_color='#8b5cf6',
            opacity=0.8
        )])
        
        fig.update_layout(
            title="Root Diameter Distribution (D10-D90)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Percentile",
            yaxis_title="Diameter (px)"
        )
        st.plotly_chart(fig, use_column_width=True)

def render_pro_dashboard_stats(stats):
    """
    Renders enhanced dashboard stats with line charts.
    """
    if not stats or not stats.get('timestamps'):
        st.info("Run some analyses to see historical trends.")
        return

    df = pd.DataFrame({
        'Time': stats['timestamps'],
        'Health': stats.get('health_scores', []),
        'Moisture': stats.get('moisture_history', []),
        'Nutrient': stats.get('nutrient_history', [])
    })
    
    # Interactive Multi-line chart
    fig = px.line(df, x='Time', y=['Health', 'Moisture', 'Nutrient'],
                  title="Historical Analysis Trends",
                  color_discrete_map={
                      'Health': '#f59e0b',
                      'Moisture': '#0ea5e9',
                      'Nutrient': '#10b981'
                  },
                  template="plotly_dark")
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_column_width=True)
