import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# CSSを定義


def add_css():
    st.markdown(
        """
        <style>
        /* 背景画像を設定 */
        .stApp {
            background-image: url('https://us.123rf.com/450wm/mindsparx/mindsparx2303/mindsparx230388323/201106478-%E8%87%AA%E5%AE%85%E3%81%AE%E3%83%99%E3%83%83%E3%83%89%E3%81%AB%E9%9D%A2%E7%99%BD%E3%81%84%E6%BC%AB%E7%94%BB%E3%81%AE%E3%82%A6%E3%82%A4%E3%83%AB%E3%82%B9%E3%82%AD%E3%83%A3%E3%83%A9%E3%82%AF%E3%82%BF%E3%83%BC%E3%80%823d%E3%83%AC%E3%83%B3%E3%83%80%E3%83%AA%E3%83%B3%E3%82%B0.jpg?ver=6'); /* 正しいパスを指定 */
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        /* タイトルのスタイル*/
        h1 {
            color: black; /* タイトルの文字色を黒に変更 */
            font-family: 'Arial', sans-serif;
            text-align: center;
        }
        /* ラジオボタンのスタイル */
        .stRadio label {
            font-size: 16px;
            color: #333333;
        }
        /* 画像の枠線 */
        img {
            border: 2px solid #4CAF50;
            border-radius: 8px;
        }

         /* ファイルアップローダーのデザインをカスタマイズ */
        .custom-file-uploader {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 20px;
            padding: 10px 20px;
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            cursor: pointer;
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }
        .custom-file-uploader:hover {
            background-color: #f5f5f5;
            border-color: #45a049;
        }
        .custom-file-uploader label {
            font-size: 16px;
            font-weight: bold;
            color: #4CAF50;
            cursor: pointer;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# YOLOv8モデルのロード
model = YOLO("model/colony_250117.pt")  # 事前トレーニング済みモデルを利用


def detect_and_count_objects(image):
    results = model(image)  # YOLOv8で画像を処理
    detections = results[0].boxes.data  # 検出結果
    num_objects = len(detections)  # オブジェクトの数をカウント

    # 検出結果を画像に描画
    for box in detections:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return image, num_objects


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
        detected_image, num_objects = detect_and_count_objects(image_np)

        st.image(detected_image, caption=f"Detected Objects: {
                 num_objects}", use_container_width=True)
        st.write(f"Number of objects detected: {num_objects}")

elif option == "Use Webcam":
    # Webカメラの映像処理
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)
    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("Webcam not found.")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_frame, num_objects = detect_and_count_objects(frame)
        FRAME_WINDOW.image(detected_frame, caption=f"Objects: {num_objects}")
    else:
        camera.release()
