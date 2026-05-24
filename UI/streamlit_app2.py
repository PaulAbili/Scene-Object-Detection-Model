import streamlit as st
import torch
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from scenes_classifier import SceneClassifier

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set models
object_classifer = YOLO('UI/object_classifer.pt')
scene_classifer = torch.load('UI/scene_classifer.pt', map_location=device, weights_only=False)

scene_classifer.to(device)
scene_classifer.eval()

# Creating output directory
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

# UI setup
st.set_page_config(layout="wide")
st.title("Scene + Object Detection Model")
st.divider()

col1, col2 = st.columns(2)

with col1:
    uploaded_file = None
    input_path = None
    ver = 0

    st.subheader("Input")

    choice = st.selectbox(
        "Choose image source",
        ["Upload an Image", "Use Sample Image"]
    )

    if choice == "Upload an Image":
        uploaded_file = st.file_uploader(
            "Upload an image ...",
            type=["jpg", "png", "jpeg", "webp", "avif"]
        )

        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")
            input_path = output_dir / uploaded_file.name
            img.save(input_path)

            st.image(img, caption="Original Image", use_container_width=True)
        else:
            st.info("Please drag, drop or upload an image to start.")

    else:
        ver = 1
        sample_images = {
            "Cows": "test_images/cows.jpg",
            "Dogs": "test_images/dogs.jpeg",
            "Person": "test_images/person.jpg",
        }

        selected_sample_image = st.selectbox(
            "Select from the sample image list",
            ["Select a Sample Image"] + list(sample_images.keys())
        )

        if selected_sample_image != "Select a Sample Image":
            input_path = sample_images[selected_sample_image]

            with open(input_path, "rb") as f:
                st.image(f.read(), caption="Original Image", use_container_width=True)

with col2:
    st.subheader("Model Output")

    if input_path is not None:
        st.text("Results will appear here when done...")
        # Image Alignment
        if ver == 0:
            st.markdown("<div style='height: 82px;'></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='height: 54px;'></div>", unsafe_allow_html=True)

        scene_results = scene_classifer.predict(
            images=[str(input_path)],
            device=device,
            topk=1
        )

        prediction = scene_results[0]
        # Only outputs top scene prediction
        top_scene = prediction['top_label']
        # Removes extra info at the start of label, rounds confidence
        st.success(
            f"Scene: {top_scene[3:]} "
            f"({prediction['top_confidence']:.2%})"
        )
        objects = object_classifer(str(input_path))
        for result in objects:
            save_path = output_dir / f"{Path(input_path).stem}_detected.jpg"

            # Adaptive box and font sizes
            h, w = result.orig_img.shape[:2]
            scale = max(h, w) / 640
            line_width = max(1, int(scale * 2))
            font_size = max(10, int(scale * 14))
            detected_image = result.plot(line_width=line_width, font_size=font_size)

            detected_image = detected_image[:, :, ::-1] # Convert from BGR to RGB
            img = Image.fromarray(detected_image)
            img.save(save_path)

        st.image(str(save_path), caption="Processed Image with Bounding Boxes", use_container_width=True)
    else:
        st.write("Results will appear here.")
