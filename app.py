import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import pandas as pd

# ---------------------------------------------------------
# Page Configuration & Metadata
# ---------------------------------------------------------
st.set_page_config(
    page_title="Handwritten Character Recognition | CNN",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern UI Styling (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.12) 50%, rgba(236, 72, 153, 0.08) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(16px);
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    .badge-primary {
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        border: 1px solid rgba(129, 140, 248, 0.3);
    }

    .badge-accuracy {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 22px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
    }

    .hero-prediction-box {
        text-align: center;
        background: linear-gradient(180deg, rgba(99, 102, 241, 0.15) 0%, rgba(30, 27, 75, 0.3) 100%);
        border: 1px solid rgba(129, 140, 248, 0.35);
        border-radius: 22px;
        padding: 30px 20px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.3);
    }

    .big-prediction-char {
        font-size: 5.5rem;
        font-weight: 800;
        line-height: 1;
        background: linear-gradient(135deg, #ffffff 0%, #c7d2fe 60%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }

    .confidence-meter {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e0e7ff;
    }

    .conf-high {
        color: #34d399;
    }

    .conf-med {
        color: #fbbf24;
    }

    .conf-low {
        color: #f87171;
    }

    .canvas-container-box {
        border: 2px dashed rgba(129, 140, 248, 0.35);
        border-radius: 20px;
        padding: 12px;
        background: rgba(10, 10, 15, 0.6);
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 12px;
    }

    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stat-char {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.2rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .stat-pct {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: #94a3b8;
    }

    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 0.82rem;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model Loading (Cached)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained CNN model...")
def load_emnist_model():
    model_path = os.path.join(os.path.dirname(__file__), "emnist_cnn_model2.keras")
    if not os.path.exists(model_path):
        st.error(f"Model file `{model_path}` not found!")
        st.stop()
    model = tf.keras.models.load_model(model_path)
    return model

model = load_emnist_model()

# ---------------------------------------------------------
# Image Preprocessing Pipeline (EMNIST Letters Centering)
# ---------------------------------------------------------
def preprocess_canvas_image(rgba_img: np.ndarray):
    """
    Preprocess user drawing to match EMNIST Letters representation:
    1. Extract grayscale values (foreground strokes).
    2. Detect stroke bounding box.
    3. Scale glyph to fit 20x20 maintaining aspect ratio.
    4. Center glyph inside 28x28 normalized canvas.
    5. Return tensor ready for CNN: shape (1, 28, 28, 1), float32 in [0, 1].
    """
    if rgba_img is None:
        return None, None

    # Grayscale extraction
    rgb = rgba_img[:, :, :3]
    alpha = rgba_img[:, :, 3]
    gray = np.max(rgb, axis=2)

    # In case user drew with opacity
    if np.max(alpha) > 0 and np.mean(alpha) < 250:
        gray = np.where(alpha > 20, gray, 0)

    # Empty canvas check
    if np.max(gray) < 25:
        return None, None

    # Detect bounding box of strokes
    coords = np.argwhere(gray > 30)
    if len(coords) == 0:
        return None, None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = gray[y0:y1, x0:x1]

    # Resize preserving aspect ratio to fit inside 20x20
    h, w = cropped.shape
    aspect = w / h
    if h > w:
        new_h = 20
        new_w = max(1, int(round(20 * aspect)))
    else:
        new_w = 20
        new_h = max(1, int(round(20 / aspect)))

    # Pillow interpolation for clean downsampling
    pil_crop = Image.fromarray(cropped.astype(np.uint8))
    pil_resized = pil_crop.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_arr = np.array(pil_resized, dtype=np.float32)

    # Center inside 28x28 frame
    canvas28 = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas28[start_y:start_y + new_h, start_x:start_x + new_w] = resized_arr

    # Normalize to [0, 1]
    normalized_tensor = canvas28 / 255.0
    input_tensor = normalized_tensor.reshape(1, 28, 28, 1)

    return input_tensor, canvas28

# ---------------------------------------------------------
# Sidebar Controls & Information
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Drawing Controls")
    stroke_width = st.slider("Stroke Thickness", min_value=10, max_value=32, value=18, step=2)
    realtime_predict = st.toggle("Real-Time Prediction", value=True, help="Update predictions immediately while drawing")
    top_k_count = st.slider("Top Predictions to Show", min_value=3, max_value=8, value=5)

    if st.button("🧹 Clear Canvas", use_container_width=True):
        st.session_state["canvas_key"] = st.session_state.get("canvas_key", 0) + 1
        st.rerun()

    st.markdown("---")
    st.markdown("### 🧠 Model Specifications")
    st.markdown("""
    - **Dataset**: EMNIST Letters (Balanced)
    - **Classes**: 26 English Alphabets (`A` - `Z`)
    - **Validation Accuracy**: `~94.08%`
    - **Input Dimensions**: `28 × 28 × 1` (Grayscale)
    - **Architecture**:
      - `Conv2D (32, 3x3, ReLU)`
      - `MaxPooling2D (2x2)`
      - `Conv2D (32, 3x3, ReLU)`
      - `MaxPooling2D (2x2)`
      - `Flatten`
      - `Dense (256, ReLU)`
      - `Dropout (0.6)`
      - `Dense (26, Softmax)`
    """)

    st.markdown("---")
    st.markdown("### 🤖 AI Usage Disclosure")
    st.info(
        "The interface styling, layout design, responsive canvas integration, and preprocessing logic "
        "were crafted with assistance from **Google DeepMind AI (Antigravity)**.",
        icon="ℹ️"
    )

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        <span class="badge-pill badge-primary">Deep Learning</span>
        <span class="badge-pill badge-primary">Convolutional Neural Network</span>
        <span class="badge-pill badge-accuracy">94.08% Validation Accuracy</span>
    </div>
    <h1 style="margin: 8px 0 6px 0; font-size: 2.2rem; font-weight: 800; color: #f8fafc;">
        Handwritten Character Recognition
    </h1>
    <p style="margin: 0; color: #94a3b8; font-size: 1.05rem;">
        Draw any uppercase English letter (<strong>A – Z</strong>) on the interactive canvas below. The CNN model will classify your stroke in real-time.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Interactive Grid
# ---------------------------------------------------------
col_canvas, col_results = st.columns([1.1, 1], gap="large")

with col_canvas:
    st.markdown("### 🎨 Interactive Drawing Canvas")
    st.caption("Use your mouse or stylus to draw an English letter. White stroke on dark background.")

    canvas_key = f"canvas_{st.session_state.get('canvas_key', 0)}"

    with st.container():
        st.markdown('<div class="canvas-container-box">', unsafe_allow_html=True)
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=canvas_key,
            update_streamlit=realtime_predict,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    manual_predict = False
    if not realtime_predict:
        manual_predict = st.button("🔮 Recognize Character", type="primary", use_container_width=True)

with col_results:
    st.markdown("### 🎯 Model Prediction")

    should_predict = realtime_predict or manual_predict

    if canvas_result is not None and canvas_result.image_data is not None and should_predict:
        input_tensor, debug_28 = preprocess_canvas_image(canvas_result.image_data)

        if input_tensor is None:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 3rem; margin-bottom: 12px;">✏️</div>
                <h4 style="color: #cbd5e1; margin-bottom: 6px;">Canvas is Empty</h4>
                <p style="color: #64748b; font-size: 0.95rem; margin: 0;">
                    Draw any letter (<strong>A–Z</strong>) on the canvas to view real-time predictions.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Predict probabilities
            probabilities = model.predict(input_tensor, verbose=0)[0]
            top_indices = np.argsort(probabilities)[::-1][:top_k_count]

            top_idx = top_indices[0]
            top_char = chr(65 + top_idx)
            top_prob = probabilities[top_idx] * 100

            # Confidence styling
            if top_prob >= 80:
                conf_class = "conf-high"
                conf_tag = "High Confidence"
            elif top_prob >= 50:
                conf_class = "conf-med"
                conf_tag = "Moderate Confidence"
            else:
                conf_class = "conf-low"
                conf_tag = "Low Confidence / Ambiguous"

            # Primary Prediction Card
            st.markdown(f"""
            <div class="hero-prediction-box">
                <div style="font-size: 0.85rem; font-weight: 700; color: #818cf8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
                    Predicted Letter
                </div>
                <div class="big-prediction-char">{top_char}</div>
                <div class="confidence-meter">
                    Confidence: <span class="{conf_class}">{top_prob:.2f}%</span>
                </div>
                <div style="margin-top: 8px;">
                    <span class="badge-pill badge-primary" style="margin: 0;">{conf_tag}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Top Candidates Breakdown
            st.markdown("#### 📊 Candidate Ranking")
            for rank, idx in enumerate(top_indices):
                letter = chr(65 + idx)
                prob = probabilities[idx] * 100
                st.markdown(f"""
                <div class="stat-row">
                    <span class="stat-char">#{rank + 1} &nbsp; <strong>{letter}</strong></span>
                    <span class="stat-pct">{prob:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(min(max(float(prob) / 100.0, 0.0), 1.0))

            # Preprocessed input inspector
            with st.expander("🔍 Inspect CNN Preprocessed Input (28×28)"):
                st.caption("This is the aspect-centered, normalized 28×28 grayscale tensor fed into the CNN.")
                st_c1, st_c2 = st.columns([1, 2])
                with st_c1:
                    st.image(debug_28, width=140, clamp=True, caption="28x28 Model Input")
                with st_c2:
                    st.markdown(f"""
                    - **Tensor Shape**: `(1, 28, 28, 1)`
                    - **Peak Pixel Value**: `{np.max(debug_28):.1f} / 255.0`
                    - **Non-zero Stroke Pixels**: `{(debug_28 > 10).sum()}`
                    - **Glyph Centering**: Aspect-preserved, 20px bounding box
                    """)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 40px 20px;">
            <div style="font-size: 3rem; margin-bottom: 12px;">✍️</div>
            <h4 style="color: #cbd5e1; margin-bottom: 6px;">Ready for Your Drawing</h4>
            <p style="color: #64748b; font-size: 0.95rem; margin: 0;">
                Draw an uppercase English letter in the canvas to recognize it.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer Section
# ---------------------------------------------------------
st.markdown("""
<div class="footer-note">
    Built with <strong>TensorFlow/Keras</strong>, <strong>Streamlit</strong>, and <strong>EMNIST Letters</strong>.
    • UI Designed with assistance from Google DeepMind AI.
</div>
""", unsafe_allow_html=True)
