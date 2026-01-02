import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import io

from backend.cnn_inference import predict_stress
from backend.root_simulator import simulate_root
from backend.ai_realism import evaluate_root_realism

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="SmartRoot-AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------
# LOAD CSS
# -------------------------------------------------
with open("static/advanced_style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------------------------------------
# AI RENDER PARAMETERS
# -------------------------------------------------
def ai_render_params(realism_score):
    return {
        "thickness_scale": min(1.4, max(0.6, 0.9 + realism_score / 500)),
        "depth_fade": min(0.9, max(0.1, 0.4 + (100 - realism_score) / 200)),
        "jitter": max(0.01, (100 - realism_score) / 300),
        "alpha": min(1.0, max(0.25, 0.85 + realism_score / 400))
    }

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🌱 SmartRoot-AI")
st.caption("CNN-based Stress Detection with AI-Enhanced Root Architecture Visualization")
st.markdown("---")

# -------------------------------------------------
# INPUTS
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "📸 Upload Vetiver Plant Image",
        ["jpg", "png", "jpeg"]
    )

with col2:
    soil_type = st.selectbox(
        "🌍 Soil Type",
        ["Sandy", "Clay", "Loamy"]
    )

# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        img_path = tmp.name

    label, moisture, nutrient = predict_stress(img_path)

    if label == "Non-Plant Image":
        st.error("❌ Please upload a valid plant image.")
        st.stop()

    st.subheader("🧠 CNN Stress Prediction")
    m1, m2, m3 = st.columns(3)
    m1.metric("Stress Type", label)
    m2.metric("Moisture (%)", moisture)
    m3.metric("Nutrients (%)", nutrient)

    st.markdown("---")

    if st.button("🌿 Simulate Root Growth", use_container_width=True):

        # -------------------------------------------------
        # SIMULATION
        # -------------------------------------------------
        segments = simulate_root(moisture, nutrient, soil_type)

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
        # VISUALIZATION
        # -------------------------------------------------
        fig, ax = plt.subplots(figsize=(4.8, 7.6), dpi=200)
        ax.axhline(y=0, color="#3b2f1e", linewidth=2)

        xs = [r["x1"] for r in segments] + [r["x2"] for r in segments]
        center_x = np.mean(xs)

        for r in segments:
            depth = min(r["y2"] / max_depth, 1.0)

            # ---- Thickness tapering (realism)
            lw = r["thickness"] * render_cfg["thickness_scale"]
            lw *= (1 - 0.6 * depth)
            lw *= (0.85 + 0.3 * np.random.rand())
            lw = max(lw, 0.6)

            # ---- Alpha depth fade
            alpha = render_cfg["alpha"] * (1 - depth * render_cfg["depth_fade"])
            alpha = min(max(alpha, 0.05), 1.0)

            # ---- Root hair zone (elongation zone only)
            is_hair = (r["thickness"] < 0.7) and (0.25 < depth < 0.75)
            if is_hair:
                lw *= 0.7
                alpha *= 0.6

            # ---- Root tip emphasis
            is_tip = (r["thickness"] < 0.5) and (depth > 0.85)
            if is_tip:
                alpha = min(alpha + 0.15, 1.0)

            # ---- Depth-based color + overlap darkening
            shade = 0.15 * depth

            r_col = 0.55 - 0.35 * depth - shade
            g_col = 0.38 - 0.25 * depth - shade
            b_col = 0.18 - 0.12 * depth
 
            #       Clamp RGB to valid range [0, 1]
            color = (
            min(max(r_col, 0.0), 1.0),
                min(max(g_col, 0.0), 1.0),
                min(max(b_col, 0.0), 1.0)
            )
            # ---- Soil resistance wobble (depth dependent)
            if is_hair:
                jx, jy = 0, 0
            else:
                soil_wobble = depth ** 2
                jx = np.random.normal(0, render_cfg["jitter"] * 0.3 * soil_wobble)
                jy = np.random.normal(0, render_cfg["jitter"] * 0.3 * soil_wobble)

            ax.plot(
                [(r["x1"] - center_x), (r["x2"] - center_x) + jx],
                [r["y1"], r["y2"] + jy],
                linewidth=lw,
                color=color,
                alpha=alpha,
                solid_capstyle="round"
            )

        ax.set_xlim(min(xs) - center_x - 0.6, max(xs) - center_x + 0.6)
        ax.set_ylim(-1, max_depth + 1)
        ax.invert_yaxis()
        ax.set_facecolor("#1b140f")
        ax.axis("off")

        # -------------------------------------------------
        # PARAMETERS ON IMAGE
        # -------------------------------------------------
        param_text = (
            f"Soil Type: {soil_type}\n"
            f"Moisture: {moisture}%\n"
            f"Nutrients: {nutrient}%"
        )

        ax.text(
            0.02, 0.92,
            param_text,
            transform=ax.transAxes,
            fontsize=9,
            color="white",
            va="top",
            ha="left",
            bbox=dict(
                facecolor="black",
                alpha=0.55,
                boxstyle="round,pad=0.4"
            )
        )

        st.subheader("🌱 AI-Enhanced Root System")
        st.pyplot(fig, use_container_width=False)

        # -------------------------------------------------
        # DOWNLOAD OPTION
        # -------------------------------------------------
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        st.download_button(
            "⬇️ Download Root Image (PNG)",
            data=buf,
            file_name="vetiver_root_ai.png",
            mime="image/png"
        )

        # -------------------------------------------------
        # ANALYTICS DISPLAY
        # -------------------------------------------------
        st.subheader("📊 Root System Analytics")

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Max Depth (cm)", round(max_depth, 2))
        a2.metric("Total Length (cm)", round(total_length, 2))
        a3.metric("Horizontal Spread (cm)", round(spread, 2))
        a4.metric("Root Hair Count", hair_count)

        # -------------------------------------------------
        # AI FEEDBACK
        # -------------------------------------------------
        st.subheader("🤖 AI Realism Assessment")
        st.metric("Realism Score", f"{realism_score}/100")

        for f in ai_feedback:
            st.write("•", f)

        st.success("✅ High-realism simulation completed")
