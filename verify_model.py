import torch
import timm
import streamlit as st
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

MODEL_PATH = "Model/thangka_convnext_fp16.pth"
IMG_SIZE = 224

st.set_page_config(page_title="Thangka AI Verification", page_icon="🖼️", layout="wide")

@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = timm.create_model(
        "convnext_tiny",
        pretrained=False,
        num_classes=num_classes
    )

    # Cast weights back to float32 natively to avoid inference precision clashes
    fp32_weights = {k: v.float() if v.dtype == torch.float16 else v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(fp32_weights)
    model.eval()

    return model, class_names

@st.cache_resource
def load_yolo_model():
    return YOLO("Model/thangka_yolo11_model_optimized_for_deployment.pt")


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict(image, model, class_names):
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_prob, top_idx = torch.max(probabilities, dim=0)

    return class_names[top_idx.item()], top_prob.item(), probabilities


model, class_names = load_model()
yolo_model = load_yolo_model()

# Header and description
st.title("🖼️ Thangka Identification System")
st.markdown("Upload a Thangka image or capture one using your camera. The model will analyze it and display its prediction.")
st.divider()

# Sidebar for inputs
st.sidebar.title("Configuration")
option = st.sidebar.radio(
    "Choose input method:",
    ["Upload Image", "Use Camera"]
)

image = None

if option == "Upload Image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Thangka image",
        type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
else:
    camera_file = st.sidebar.camera_input("Capture image")
    if camera_file is not None:
        image = Image.open(camera_file)

# Main container for displaying results
if image is not None:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### 📸 Input Image")
        # Limiting width internally keeping it smaller compared to full width
        st.image(image, caption="Captured/Uploaded Image", use_container_width=True)
        
    with col2:
        st.markdown("### 📊 Dual Model Analysis")
        with st.spinner("Analyzing the image with both models..."):
            predicted_class, confidence, probabilities = predict(
                image,
                model,
                class_names
            )
            yolo_results = yolo_model(image)
            yolo_res = yolo_results[0]
            if hasattr(yolo_res, 'probs') and yolo_res.probs is not None:
                yolo_top1_idx = yolo_res.probs.top1
                yolo_conf = yolo_res.probs.top1conf.item()
                yolo_pred_class = yolo_model.names[yolo_top1_idx]
                yolo_probs = yolo_res.probs.data
            else:
                yolo_pred_class = "Unknown"
                yolo_conf = 0.0
                yolo_probs = None
                
        mod_col1, mod_col2 = st.columns(2)
        
        with mod_col1:
            st.markdown("#### ConvNeXt")
            st.metric(label="Prediction", value=predicted_class, delta=f"{confidence * 100:.2f}% conf", delta_color="normal")
                
        with mod_col2:
            st.markdown("#### YOLO")
            st.metric(label="Prediction", value=yolo_pred_class, delta=f"{yolo_conf * 100:.2f}% conf", delta_color="normal")
else:
    st.info("👈 Please select an input method from the sidebar and provide an image to see the analysis.")