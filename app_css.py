import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from collections import Counter

# CSSを定義
def add_css():
    st.markdown(
        """
        <style>
        /* 背景画像を設定 */
        .stApp {
            background-image: url('');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        /* カスタムグループコンテナ */
        .custom-group-container {

        }
        </style>
        """,
        unsafe_allow_html=True
    )


# YOLOv8モデルのロード
model = YOLO("model/colony_241126.pt")  # 事前トレーニング済みモデルを利用

# 検出関数
def detect_and_count_objects(image):
    results = model(image)  # YOLOv8で画像を処理
    detections = results[0].boxes.data  # 検出結果

    # 種類別のカウント
    class_counts = Counter()

    # 検出結果を画像に描画
    for box in detections:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{model.names[int(cls)]} {conf:.2f}"
        class_counts[model.names[int(cls)]] += 1  # クラスのカウント

        # バウンディングボックスとラベルを描画
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,  # フォントサイズを調整
                    (255, 0, 0), 2)  # 太さを調整

    return image, class_counts


# CSSを適用
add_css()

# Streamlit UI
st.title("Colony Object Counter")
st.write("Upload an image or use the webcam to detect and count objects.")

# 画像アップロードまたはWebカメラ選択
option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        detected_image, class_counts = detect_and_count_objects(image_np)

        # カスタムHTMLコンテナでグループ化
        st.markdown('<div class="custom-group-container">',
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.image(detected_image, caption="Detection Results",
                     use_container_width=True)

        with col2:
            st.write("**Detected Object Counts:**")
            for obj_class, count in class_counts.items():
                st.write(f"- {obj_class}: {count}")
        st.markdown('</div>', unsafe_allow_html=True)


elif option == "Use Webcam":
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)
    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("Webcam not found.")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_frame, class_counts = detect_and_count_objects(frame)

        # カスタムHTMLコンテナでグループ化
        st.markdown('<div class="custom-group-container">',
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            FRAME_WINDOW.image(detected_frame, caption="Live Detection")

        with col2:
            st.write("**Detected Object Counts:**")
            for obj_class, count in class_counts.items():
                st.write(f"- {obj_class}: {count}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        camera.release()
