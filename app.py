import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import base64
import pandas as pd
from datetime import datetime

# UI


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded


encoded_header = get_base64_image("images/colony-y.png")
encoded_bk = get_base64_image("images/colony-sea-pre.png")

st.set_page_config(
    page_title="Colony Counter",
    page_icon="images/icon_1.png",
)

header_html = f'''
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded_header}" alt="Title Image" />
    </div>
'''
st.markdown(header_html, unsafe_allow_html=True)

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

        [data-testid="stHorizontalBlock"]{{
            padding:20px;
            background-color: black;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


add_css()

try:
    model = YOLO("model/best_train21.pt")
except Exception as e:
    st.error(f"Error loading model: {e}")

CONFIDENCE_THRESHOLD = 0.10

class_mapping = {
    "E": "E.coli",
    "Y": "S.aureus",
    "S": "S.cerevisiae"
}

color_mapping = {
    "E.coli": (0, 255, 0),
    "S.aureus": (244, 229, 17),
    "S.cerevisiae": (255, 0, 0)
}


def detect_and_count_objects(image):
    results = model(image)
    detections = results[0].boxes.data

    class_counts = Counter()
    overlay = image.copy()

    for box in detections:
        x1, y1, x2, y2, conf, cls = box
        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        raw_class = model.names[int(cls)]
        display_class = class_mapping.get(raw_class, raw_class)
        label = f"{display_class} {conf:.2f}"

        class_counts[display_class] += 1

        color = color_mapping.get(display_class, (0, 255, 0))

        # 塗りつぶし矩形（透明度を上げる）
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # 画像と半透明矩形をブレンド
    alpha = 0.2  # 透明度を上げる（0.2 = ほぼ透明）
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # ラベルを直接描画（オブジェクトの色を反映）
    for box in detections:
        x1, y1, x2, y2, conf, cls = box
        if conf < CONFIDENCE_THRESHOLD:
            continue
        x1, y1 = map(int, [x1, y1])
        raw_class = model.names[int(cls)]
        display_class = class_mapping.get(raw_class, raw_class)
        label = f"{display_class} {conf:.2f}"
        label_color = color_mapping.get(
            display_class, (0, 255, 0))  # ラベルの色もオブジェクトの色に

        cv2.putText(image, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

    return image, class_counts


option = st.radio(
    "Choose input method:",
    ("📁 Upload Image", "📷 Use Webcam")
)


def save_to_csv(class_counts, filename="detected_objects.csv"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = {
        "Timestamp": [timestamp],
        "S.aureus": [class_counts.get("S.aureus", 0)],
        "S.cerevisiae": [class_counts.get("S.cerevisiae", 0)],
        "E.coli": [class_counts.get("E.coli", 0)],
        "Total": [sum(class_counts.values())]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, mode='a', header=not pd.read_csv(
        filename).empty if pd.io.common.file_exists(filename) else True, index=False)


if option == "📁 Upload Image":
    uploaded_file = st.file_uploader(
        "📂 画像を選択してください...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        detected_image, class_counts = detect_and_count_objects(image_np)

        col1, col2 = st.columns(2)
        with col1:
            st.image(detected_image, caption="検出結果", use_container_width=True)
        with col2:
            st.write("🧐 **検出されたオブジェクトの数:**")
            st.write(f"🦠 S.aureus: {class_counts.get('S.aureus', 0)}")
            st.write(f"🧪 S.cerevisiae: {class_counts.get('S.cerevisiae', 0)}")
            st.write(f"🧬 E.coli: {class_counts.get('E.coli', 0)}")
            st.write(f"📊 **総数: {sum(class_counts.values())}**")

        save_to_csv(class_counts)

elif option == "📷 Use Webcam":
    captured_image = st.camera_input("📸 デバイスのカメラで撮影")
    if captured_image is not None:
        image = Image.open(captured_image).convert("RGB")
        image_np = np.array(image)

        detected_image, class_counts = detect_and_count_objects(image_np)

        col1, col2 = st.columns(2)
        with col1:
            st.image(detected_image, caption="撮影された写真",
                     use_container_width=True)
        with col2:
            st.write("🔍 **検出されたオブジェクトの数:**")
            st.write(f"🦠 S.aureus: {class_counts.get('S.aureus', 0)}")
            st.write(f"🧪 S.cerevisiae: {class_counts.get('S.cerevisiae', 0)}")
            st.write(f"🧬 E.coli: {class_counts.get('E.coli', 0)}")
            st.write(f"📊 **総数: {sum(class_counts.values())}**")

        save_to_csv(class_counts)
