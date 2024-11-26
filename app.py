import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# YOLOv8モデルのロード
model = YOLO("model/colony_600.pt")  # 事前トレーニング済みモデルを利用

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
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return image, num_objects

# Streamlit UI
st.title("Colony Object Counter")
st.write("Upload an image or use the webcam to detect and count objects.")

# 画像アップロードまたはWebカメラ選択
option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        detected_image, num_objects = detect_and_count_objects(image_np)

        st.image(detected_image, caption=f"Detected Objects: {num_objects}", use_container_width=True)
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
