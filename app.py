import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import base64
import pandas as pd  # pandasを追加
from datetime import datetime  # タイムスタンプ用


# UI

# アイコンのBase64エンコード


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded


# ヘッダー用アイコン
encoded_header = get_base64_image("images/colony-y.png")
encoded_bk = get_base64_image("images/colony-sea-pre.png")

# ファビコンの設定
st.set_page_config(
    page_title="Colony Counter",  # タイトル
    page_icon="images/icon_1.png",  # Unicode絵文字 または アイコン画像のパス
)

# アイコン付きヘッダー
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

        [data-testid="stHorizontalBlock"]{{
            padding:20px;
            background-color: black;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# CSSを適用
add_css()


# YOLOv8モデルのロード
try:
    model = YOLO("model/best_0206_1611.pt")
except Exception as e:
    st.error(f"Error loading model: {e}")

# 検出の信頼度の閾値（例: 0.5 以上のものだけを処理）
CONFIDENCE_THRESHOLD = 0.3

# クラス名の変換用辞書
class_mapping = {
    "E": "E.coli",
    "Y": "S.aureus",
    "S": "S.cerevisiae"
}


# 色のマッピング（例：クラスごとに異なる色を指定）
color_mapping = {
    "E.coli": (0, 255, 0),  # Green for Escherichia
    "S.aureus": (244, 229, 17),  # Red for Staphylococcus
    "S.cerevisiae": (255, 0, 0)  # Blue for S. cerevisiae
}

# 検出関数


def detect_and_count_objects(image):
    results = model(image)  # YOLOv8で画像を処理
    detections = results[0].boxes.data  # 検出結果

    # 種類別のカウント（後で表示するため日本語に変換したキーでカウント）
    class_counts = Counter()

    # 検出結果を画像に描画
    for box in detections:
        x1, y1, x2, y2, conf, cls = box
        # 信頼度が閾値未満の場合はスキップ
        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        raw_class = model.names[int(cls)]
        display_class = class_mapping.get(raw_class, raw_class)  # 変換後のクラス名を使用
        # 修正ポイント：raw_class ではなく display_class を使用
        label = f"{display_class} {conf:.2f}"

        # カウント更新
        class_counts[display_class] += 1

        # クラスごとに異なる色を適用
        color = color_mapping.get(display_class, (0, 255, 0))  # デフォルトは緑

        # 画像に矩形を描画
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        # クラス名と信頼度を画像にテキストとして表示
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    color, 2)

    return image, class_counts


# 画像アップロードまたはWebカメラ選択
# オプション選択
option = st.radio(
    "Choose input method:",
    ("📁 Upload Image", "📷 Use Webcam")  # アイコン付き
)

# 検出処理

# 検出データをCSVに保存する関数


def save_to_csv(class_counts, filename="detected_objects.csv"):
    # 現在の日時を取得
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 各菌のカウントを取得（未検出の場合は0）
    s_aureus_count = class_counts.get("S.aureus", 0)
    s_cerevisiae_count = class_counts.get("S.cerevisiae", 0)
    e_coli_count = class_counts.get("E.coli", 0)
    total_count = s_aureus_count + s_cerevisiae_count + e_coli_count

    # データフレームを作成
    data = {
        "Timestamp": [timestamp],
        "S.aureus": [s_aureus_count],
        "S.cerevisiae": [s_cerevisiae_count],
        "E.coli": [e_coli_count],
        "Total": [total_count]
    }
    df = pd.DataFrame(data)

    # CSVに変換してBase64でエンコード
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()

    # ダウンロードリンクを作成
    href = f'<a href="data:file/csv;base64,{
        b64}" download="{filename}">📥 CSVファイルをダウンロード</a>'
    return href


# 検出処理の部分にCSV保存機能を追加
if option == "📁 Upload Image":
    uploaded_file = st.file_uploader(
        "📂 画像を選択してください...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # 検出処理
        detected_image, class_counts = detect_and_count_objects(image_np)

        col1, col2 = st.columns(2)
        with col1:
            st.image(detected_image, caption="検出結果", use_container_width=True)
        with col2:
            st.write("🧐 **検出されたオブジェクトの数:**")
            st.write(f"🦠 S.aureus: {class_counts.get('S.aureus', 0)}")
            st.write(f"🧫 S.cerevisiae: {class_counts.get('S.cerevisiae', 0)}")
            st.write(f"🧬 E.coli: {class_counts.get('E.coli', 0)}")
            total_count = sum(class_counts.values())
            st.write(f"📊 **総数: {total_count}**")

        # CSVダウンロードリンクを表示
        st.markdown(save_to_csv(class_counts), unsafe_allow_html=True)

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
            st.write(f"🧫 S.cerevisiae: {class_counts.get('S.cerevisiae', 0)}")
            st.write(f"🧬 E.coli: {class_counts.get('E.coli', 0)}")
            total_count = sum(class_counts.values())
            st.write(f"📊 **総数: {total_count}**")

        # CSVダウンロードリンクを表示
        st.markdown(save_to_csv(class_counts), unsafe_allow_html=True)
