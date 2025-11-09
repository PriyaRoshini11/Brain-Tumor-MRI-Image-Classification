import streamlit as st
import tensorflow as tf
import pandas as pd
from keras.preprocessing import image
import numpy as np
import json
from PIL import Image
import plotly.express as px

# -----------------------------
# ⚙️ App Configuration
# -----------------------------
st.set_page_config(page_title="🧠 Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")
with st.sidebar:
    st.header("📘 Project Info")
    st.markdown("""
    **Brain Tumor MRI Image Classification**
    - Models: CNN, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0  
    - Final model: **InceptionV3 (Fine-tuned)**  
    - Trained on 1,695 MRI images
    - Classes: Glioma, Meningioma, Pituitary, No Tumor
    """)
st.title("🧠 Brain Tumor MRI Image Classification")
st.markdown("Upload an MRI image, and the model will predict the tumor type!")

# -----------------------------
# 🔍 Load Model and Labels
# -----------------------------
@st.cache_resource
def load_brain_model():
    model = tf.keras.models.load_model("final_best_model.keras")  # best model saved automatically
    with open("class_indices.json", "r") as f:
        labels_dict = json.load(f)
    # convert to list ordered by index
    class_labels = [labels_dict[str(i)] if str(i) in labels_dict else labels_dict[i] for i in sorted(map(int, labels_dict.keys()))]
    return model, class_labels

model, class_labels = load_brain_model()
st.success("✅ Model and class labels loaded successfully!")

# -----------------------------
# 📤 Upload Section
# -----------------------------
uploaded_file = st.file_uploader("📸 Upload a Brain MRI image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🧠 Uploaded MRI Image", use_container_width=True)

    # -----------------------------
    # 🧠 Preprocess Image
    # -----------------------------
    img_resized = img.resize((224, 224))  # match model input
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------
    # 🔮 Model Prediction
    # -----------------------------
    preds = model.predict(img_array)
    preds_percent = preds[0] * 100
    pred_index = np.argmax(preds[0])
    pred_label = class_labels[pred_index]
    confidence = preds_percent[pred_index]

    # -----------------------------
    # 🏁 Display Results
    # -----------------------------
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    st.write(f"**🧠 Predicted Tumor Type:** {pred_label}")
    st.write(f"**💪 Confidence:** {confidence:.2f}%")

    # Plotly bar visualization
    prob_df = pd.DataFrame({
        "Tumor Category": class_labels,
        "Confidence (%)": preds_percent
    }).sort_values("Confidence (%)", ascending=False)

    fig = px.bar(
        prob_df,
        x="Tumor Category",
        y="Confidence (%)",
        color="Confidence (%)",
        color_continuous_scale="Purples",
        text_auto=".2f",
        title="Predicted Confidence per Tumor Type"
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Tumor Category",
        yaxis_title="Confidence (%)",
        xaxis_tickangle=-30,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12)
    )

    st.plotly_chart(fig, use_container_width=True)

    if confidence < 50:
        st.warning("⚠️ The model is not highly confident. The image may be unclear or could belong to an ambiguous category.")
else:
    st.info("👆 Upload a Brain MRI image to begin classification.")

with st.expander("📘 Learn about the Tumor Categories"):
    st.markdown("""
    ### 🧬 **Class Information**
    **🩸 Glioma Tumor**
    - Develops from glial cells that support neurons.  
    - Appears as irregular gray or white mass on MRI.  
    - Common symptoms: headaches, seizures, nausea.  

    **🧠 Meningioma Tumor**
    - Forms on brain’s protective layer (meninges).  
    - Usually benign but can grow large.  
    - Treatment: surgical removal or observation.  

    **🩺 Pituitary Tumor**
    - Small growth in pituitary gland (controls hormones).  
    - May cause vision or hormonal changes.  
    - Often non-cancerous.  

    **✅ No Tumor (Normal MRI)**
    - Healthy brain with no abnormal tissue.  
    - Acts as baseline for comparison.
    """)

# -----------------------------
# 📚 App Footer
# -----------------------------
st.markdown("---")
st.caption("Developed by **S. Priya Roshini** 🧠 using TensorFlow & Streamlit | 2025")