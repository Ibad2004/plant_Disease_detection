import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# ----------------------------
# 🧠 Define your CNN model
# ----------------------------
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.35)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 56, 56]
        x = self.pool(F.relu(self.conv3(x)))  # [B, 128, 28, 28]
        x = self.pool(F.relu(self.conv4(x)))  # [B, 256, 14, 14]
        x = x.view(-1, 256 * 14 * 14)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ----------------------------
# 🏷️ Class Labels
# ----------------------------
class_names = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot", "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus", "Tomato_healthy"
]

# ----------------------------
# 🧠 Load Model
# ----------------------------
@st.cache_resource
def load_model():
    model = PlantDiseaseCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load("plant_disease_cnn.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# ----------------------------
# 🔄 Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ----------------------------
# 🌿 Streamlit UI
# ----------------------------
st.set_page_config(page_title="Plant Disease Classifier", layout="centered")

st.title("🌿 Plant Disease Detection App")
st.caption("Upload a leaf image to detect plant disease with confidence levels.")

tab1, tab2 = st.tabs(["🔍 Predict", "📊 About Model"])

with tab1:
    uploaded_file = st.file_uploader("📁 Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        top_probs, top_indices = torch.topk(probabilities, 5)
        top_classes = [class_names[i] for i in top_indices]

        st.markdown("### 🧠 Top Predictions:")
        results_df = pd.DataFrame({
            "Class": top_classes,
            "Confidence (%)": (top_probs.numpy() * 100).round(2)
        })

        st.dataframe(results_df, use_container_width=True)

        # 🎯 Visualization
        st.markdown("### 📈 Confidence Bar Chart")
        fig, ax = plt.subplots()
        ax.barh(top_classes[::-1], top_probs.numpy()[::-1] * 100, color='seagreen')
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Top 5 Predictions")
        st.pyplot(fig)

        # 💾 Download
        csv = results_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction_results.csv">📥 Download Predictions as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

with tab2:
    st.subheader("Model Architecture")
    st.code("4 Convolution Layers → Dropout → Fully Connected → Softmax")

    st.subheader("Preprocessing Used")
    st.markdown("- Resize to **224x224**")
    st.markdown("- Normalize to mean **0.5** and std **0.5** per channel")
    st.markdown("- RGB → Tensor")

    st.info("Model was trained on ~80,000 images and achieves ~95% validation accuracy.")
