
import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np

from PIL import Image
from torchvision import transforms, models
from pathlib import Path

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "resnet18_baseline_best.pth"

THRESHOLD = 0.695665
DEVICE = torch.device("cpu")

st.set_page_config(
    page_title="Iris Presentation Attack Detection",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ Iris Presentation Attack Detection")

st.write(
    "Upload an iris image or capture an image using your webcam "
    "to run the presentation attack detection model."
)

# --------------------------------------------------
# 2. Load trained ResNet18 model
# --------------------------------------------------

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# 3. Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 4. Prediction function
# --------------------------------------------------

def predict_image(image):

    model = load_model()

    image = image.convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    attack_score = probabilities[0][0].item()
    bonafide_score = probabilities[0][1].item()

    return attack_score, bonafide_score


# --------------------------------------------------
# 5. Display prediction
# --------------------------------------------------

def display_prediction(attack_score, bonafide_score):

    st.subheader("Classification Result")

    st.metric(
        "Bona fide score",
        f"{bonafide_score:.4f}"
    )

    st.metric(
        "Attack score",
        f"{attack_score:.4f}"
    )

    if bonafide_score >= THRESHOLD:

        st.success("Classification: BONA FIDE")

    else:

        st.warning("Classification: PRINTED IRIS ATTACK")

    st.caption(f"Decision threshold: {THRESHOLD:.6f}")

    st.caption(
        "Research demonstration only. This prediction does not "
        "prove identity, liveness, or secure authentication."
    )


# --------------------------------------------------
# 6. Upload iris image
# --------------------------------------------------

st.header("1. Upload Iris Image")

uploaded_file = st.file_uploader(
    "Choose an iris image",
    type=["png", "jpg", "jpeg", "bmp"],
    key="iris_upload"
)

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)
        image.verify()

        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded iris image",
            width=350
        )

        attack_score, bonafide_score = predict_image(image)

        display_prediction(attack_score, bonafide_score)

    except Exception as error:

        st.error(f"Upload prediction error: {error}")


# --------------------------------------------------
# 7. Webcam eye detection
# --------------------------------------------------

st.divider()

st.header("2. Live Camera Eye Detection")

st.write(
    "Position your eye in front of the camera and capture an image."
)

camera_file = st.camera_input(
    "Capture your eye",
    key="iris_camera"
)

if camera_file is not None:

    try:

        # Read webcam image
        camera_image = Image.open(camera_file).convert("RGB")

        # Convert to OpenCV format
        frame = np.array(camera_image)

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )

        # Load eye detection classifier
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        if eye_cascade.empty():

            st.error("Unable to load eye detection classifier.")
            st.stop()

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(30, 30)
        )

        # Sort detections by area, largest first
        eyes = sorted(
            eyes,
            key=lambda eye: eye[2] * eye[3],
            reverse=True
        )

        # Draw red rectangles
        result_frame = frame.copy()

        for (x, y, w, h) in eyes:

            cv2.rectangle(
                result_frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                3
            )

        st.image(
            result_frame,
            caption="Detected eye regions",
            use_container_width=True
        )

        # Process detected eye
        if len(eyes) > 0:

            st.success(
                f"Detected {len(eyes)} possible eye region(s)."
            )

            # Select largest detected region
            x, y, w, h = eyes[0]

            # Crop eye from original RGB image
            eye_crop = camera_image.crop(
                (x, y, x + w, y + h)
            )

            st.subheader("Detected Eye Crop")

            st.image(
                eye_crop,
                caption="Eye region selected for model processing",
                width=300
            )

            # Run trained ResNet18 model
            st.subheader("Running Iris PAD Model...")

            attack_score, bonafide_score = predict_image(
                eye_crop
            )

            display_prediction(
                attack_score,
                bonafide_score
            )

        else:

            st.warning(
                "No eyes detected. Improve lighting and move closer "
                "to the camera before capturing another image."
            )

    except Exception as error:

        st.error(
            f"Camera processing error: {error}"
        )
