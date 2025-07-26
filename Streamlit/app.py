import streamlit as st
from PIL import Image
from clf import predict_with_model

st.title("Fracture Detection with Multiple Models")

# Chọn mô hình
model_choice = st.selectbox("Choose a model", [
    "ViT (Vision Transformer)",
    "ResNet18",
    "MobileNetV2",
    "DenseNet121"
])

file_up = st.file_uploader("Upload a grayscale X-ray image", type=["jpg", "jpeg", "png"])

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button('Predict'):
        label = predict_with_model(image, model_choice)
        if label == 0:
            st.write("🔴 **The image is: Fractured**")
        else:
            st.write("🟢 **The image is: Not Fractured**")
else:
    st.write("Please upload an image.")
