import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import base64

# ローカル画像をBase64エンコードする関数


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded


# ローカル背景画像をエンコード
encoded_bk = get_base64_image("images/colony-sea-pre.png")

# CSSを定義（より具体的なセレクタを使用）


def add_css():
    st.markdown(
        f"""
        <style>
        /* Streamlit のメインコンテナに背景を設定する */
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded_bk}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# YOLOv8モデルのロード
try:
    model = YOLO("model/best_0130_2.pt")
except Exception as e:
    st.error(f"Error loading model: {e}")

# 検出の信頼度の閾値（例: 0.5 以上のものだけを処理）
CONFIDENCE_THRESHOLD = 0.5

# クラス名の変換用辞書
class_mapping = {
    "E": "大腸菌",
    "Y": "黄色ブドウ球菌",
    "S": "酵母"
}

# 検出関数


def detect_and_count_objects(image):
    results = model(image)  # YOLOv8で画像を処理
    detections = results[0].boxes.data  # 検出結果

    # 種類別のカウント（後で表示するため日本語に変換したキーでカウント）
    class_counts = Counter()

    # 検出結果を画像に描画（ラベルは元のクラス名 "E", "Y", "S" のまま）
    for box in detections:
        x1, y1, x2, y2, conf, cls = box
        # 信頼度が閾値未満の場合はスキップ
        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        raw_class = model.names[int(cls)]
        label = f"{raw_class} {conf:.2f}"
        display_class = class_mapping.get(raw_class, raw_class)
        class_counts[display_class] += 1

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 0, 0), 2)
    return image, class_counts


# CSSを適用
add_css()

# ヘッダー用のローカル画像をエンコード（例：images/colony-title3.png）
encoded_header = get_base64_image("images/colony-title3.png")
header_html = f'''
<h1>
    <img style="margin-right: 10px;" src="data:image/png;base64,{encoded_header}" alt="virus" />
</h1>
'''
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

        st.markdown('<div class="custom-group-container">',
                    unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

elif option == "Use Webcam":
    # st.camera_input を利用してカメラから画像をキャプチャ
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
