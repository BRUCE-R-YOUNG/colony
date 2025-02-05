import streamlit as st
import cv2
import numpy as np
import requests
import base64
from PIL import Image
from collections import Counter

# RoboflowのAPI設定
ROBOFLOW_API_KEY = "a14bfchqXiktu80vG8QM"
WORKSPACE = "iotbio-rksli"  # ワークスペース名
PROJECT = "colony-pplnw"  # プロジェクト名
VERSION = "12"  # モデルのバージョン

# ローカル画像をBase64エンコードする関数


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded


# ローカル背景画像をエンコード
encoded_bk = get_base64_image("images/colony-sea-pre.png")

# StreamlitのUIカスタマイズ
hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def add_css():
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded_bk}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# 検出の信頼度の閾値
CONFIDENCE_THRESHOLD = 0.5

# クラス名の変換用辞書
class_mapping = {
    "E": "E.coli",
    "Y": "S.aureus",
    "S": "S.cerevisiae"
}

# 色のマッピング（例：クラスごとに異なる色を指定）
color_mapping = {
    "E.coli": (0, 255, 0),  # Green for Escherichia
    "S.aureus": (244, 229, 17),  # Yellow for Staphylococcus
    "S.cerevisiae": (255, 0, 0)  # Red for S. cerevisiae
}

# Roboflow APIを使った物体検出関数


def detect_and_count_objects(image):
    _, img_encoded = cv2.imencode(".jpg", image)
    img_bytes = img_encoded.tobytes()

    response = requests.post(
        f"https://detect.roboflow.com/{PROJECT}/{
            VERSION}?api_key={ROBOFLOW_API_KEY}",
        files={"file": ("image.jpg", img_bytes, "image/jpeg")}
    )

    if response.status_code != 200:
        st.error(f"Error in Roboflow API request: {response.text}")
        return image, {}

    detections = response.json().get("predictions", [])
    class_counts = Counter()

    for detection in detections:
        x, y, w, h = int(detection["x"]), int(detection["y"]), int(
            detection["width"]), int(detection["height"])
        class_name = detection["class"]
        confidence = detection["confidence"]

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        display_class = class_mapping.get(class_name, class_name)
        class_counts[display_class] += 1
        color = color_mapping.get(display_class, (0, 255, 0))

        cv2.rectangle(image, (x - w // 2, y - h // 2),
                      (x + w // 2, y + h // 2), color, 2)
        cv2.putText(image, f"{display_class} {confidence:.2f}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return image, class_counts


# CSSを適用
add_css()

# ヘッダー用のローカル画像をエンコード
encoded_header = get_base64_image("images/colony-y.png")
header_html = f'<img src="data:image/png;base64,{
    encoded_header}" alt="Header Image" />'
st.markdown(header_html, unsafe_allow_html=True)

# 画像アップロードまたはWebカメラ選択
option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        detected_image, class_counts = detect_and_count_objects(image_np)

        col1, col2 = st.columns(2)
        with col1:
            st.image(detected_image, caption="Detection Results",
                     use_container_width=True)
        with col2:
            st.write("**Detected Object Counts:**")
            total_count = sum(class_counts.values())
            for obj_class, count in class_counts.items():
                st.write(f"- {obj_class}: {count}")
            st.write(f"**Total Objects: {total_count}**")

elif option == "Use Webcam":
    captured_image = st.camera_input("Capture Photo using your device")
    if captured_image is not None:
        image = Image.open(captured_image).convert("RGB")
        image_np = np.array(image)
        detected_image, class_counts = detect_and_count_objects(image_np)

        col1, col2 = st.columns(2)
        with col1:
            st.image(detected_image, caption="Captured Photo",
                     use_container_width=True)
        with col2:
            st.write("**Detected Object Counts:**")
            total_count = sum(class_counts.values())
            for obj_class, count in class_counts.items():
                st.write(f"- {obj_class}: {count}")
        st.write(f"**Total Objects: {total_count}**")
