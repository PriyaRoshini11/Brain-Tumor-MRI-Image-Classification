# Brain-Tumor-MRI-Image-Classification

**Problem Statement:**

This project aims to develop a deep learning-based solution for classifying brain MRI images into multiple categories(Glioma, Meningioma, Pituitary, and No Tumor) according to tumor type. 

It involves building a custom CNN model from scratch and enhancing performance through transfer learning using pretrained models.(ResNet50, MobileNetV2, InceptionV3, EfficientNetB0)

The project also includes deploying a user-friendly Streamlit web application to enable real-time tumor type predictions from uploaded MRI images.

**🌍 Domain:**

Medical Imaging · Brain MRI Classification

**🧠 Skills Gained:**

Image Augmentation & Preprocessing

Deep Learning & Transfer Learning

Model Fine-Tuning

Model Evaluation & Visualization

Streamlit App Deployment

Python, TensorFlow/Keras

**📌 Real-Time Business Use Cases:**
**1.	AI-Assisted Medical Diagnosis:**
 Provide radiologists with AI-powered tools to quickly classify brain tumors based on MRI images, reducing diagnostic turnaround time and increasing accuracy.

**2.	Early Detection and Patient Triage:**
 Automatically flag high-risk MRI images for immediate specialist review, improving hospital workflow and patient care prioritization.

**3.	Research and Clinical Trials:**
 Use AI classification tools to segment patient datasets by tumor type, aiding in research studies and clinical trial recruitment.

**4.	Second-Opinion AI Systems:**
 Deploy AI-powered classification tools in telemedicine or remote consultation setups for second-opinion diagnostics in under-resourced healthcare regions.

** 🧠 Technologies Used:**

| Category                   | Tools / Libraries                       |
| -------------------------- | --------------------------------------- |
| 🐍 Programming             | Python                                  |
| 🧠 Deep Learning           | TensorFlow, Keras                       |
| 📊 Visualization           | Matplotlib, Seaborn, Plotly             |
| 🧮 Evaluation              | Scikit-learn                            |
| 🖼️ Image Handling          | Pillow (PIL), ImageDataGenerator        |
| 🌐 Deployment              | Streamlit                               |
| 💻 Development Environment | Google Colab, Jupyter Notebook, VS Code |

**🚀 Deployment (Streamlit App):**

File: brain_streamlit.py

**Features:**

Upload MRI images (.jpg, .jpeg, .png)

Predict tumor type (Glioma, Meningioma, Pituitary, or No Tumor)

View model confidence percentage

Interactive bar chart visualization for confidence levels

Informative sidebar with tumor class descriptions and project info
