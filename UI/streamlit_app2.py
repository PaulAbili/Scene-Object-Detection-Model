import streamlit as st
from PIL import Image
import time

from ultralytics import YOLO
from pathlib import Path

import torch
from scenes_classifier import SceneClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"
# Models
object_classifer = YOLO('object_classifer.pt')
scene_classifer = torch.load('scene_classifer.pt', map_location=device, weights_only=False)

scene_classifer.to(device)
scene_classifer.eval()

# Creating output directory
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

st.set_page_config(layout="wide") 
st.title("Object + Scene Detection Model")
st.divider()

# Create two columns for the "Drag and Drop" vs "Output" look
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    
    input_slot = st.empty()

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        input_path = output_dir / uploaded_file.name
        img.save(input_path)

        # Display the uploaded image
        st.image(img, caption="Original Image", use_container_width=True)
    else:
        # Placeholder box if no image is uploaded
        st.info("Please drag and drop or upload an image to start.")

with col2:
    st.subheader("Model Output") 
    
    if uploaded_file is not None:
        st.text("Results will appear here when done...")
        scene_results = scene_classifer.predict(
            images=[str(input_path)],
            device=device,
            topk=1
        )

        prediction = scene_results[0]
        out = prediction['top_label']
        st.success(
            f"Scene: {out[3:]} "
            f"({prediction['top_confidence']:.2%})"
        )
        objects = object_classifer(str(input_path))
        for result in objects:
            save_path = output_dir / f"{input_path.stem}_detected.jpg"
            result.save(filename=str(save_path))

        st.image(str(save_path), caption="Processed Image with Bounding Boxes", use_container_width=True)
       

    else:
        st.write("Results will appear here.")
