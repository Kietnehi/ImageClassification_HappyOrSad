import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt  # Dùng Matplotlib để hiển thị ảnh
from PIL import Image  # Import Pillow để sử dụng Image.open()
import base64
from io import BytesIO
    
# Thư mục chứa hình ảnh cần dự đoán
image_folder = './test'

def img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
# Sau khi huấn luyện, bạn có thể tải lại mô hình đã lưu
model = load_model('my_model.keras')  # Tải lại mô hình từ file .h5

# Kích thước ảnh chuẩn hóa
IMG_SIZE = 150

# Hàm đọc và chuẩn hóa các hình ảnh trong thư mục
def prepare_images(image_folder):
    images = []
    filenames = []

    for img_name in os.listdir(image_folder):  # Lặp qua các tệp trong thư mục
        img_path = os.path.join(image_folder, img_name)

        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Thay đổi kích thước ảnh
            img = img / 255.0  # Chuẩn hóa ảnh (giới hạn từ 0 đến 1)
            images.append(img)
            filenames.append(img_name)

    # Chuyển đổi danh sách ảnh thành mảng NumPy
    return np.array(images), filenames


def predict_and_display_images(image_folder, model, batch_size=10):
    # Đọc và chuẩn hóa tất cả ảnh trong thư mục
    images, filenames = prepare_images(image_folder)

    # Chia thành các batch nhỏ
    num_batches = len(images) // batch_size + (1 if len(images) % batch_size > 0 else 0)

    # Dự đoán theo từng batch
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(images))  # Đảm bảo không vượt quá số lượng hình ảnh

        # Lấy batch ảnh cần dự đoán
        batch_images = images[start_idx:end_idx]

        # Dự đoán trên batch
        predictions = model.predict(batch_images)

        # Hiển thị kết quả dự đoán trên các hình ảnh trong batch
        for i, filename in enumerate(filenames[start_idx:end_idx]):
            prediction = predictions[i]
            label = "Vui" if prediction[0] < 0.5 else "Buồn"
            confidence = (1 - prediction[0]) * 100 if label == "Vui" else prediction[0] * 100  # Tỉ lệ xác suất

            # In ra kết quả
            st.write(f"{filename}: {label} ({confidence:.2f}%)")

            # Đọc lại hình ảnh gốc để hiển thị
            img = cv2.imread(os.path.join(image_folder, filename))

            # Thêm văn bản vào ảnh
            text = f"{label}: {confidence:.2f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = (10, 30)  # Vị trí của văn bản
            font_scale = 0.5
            color = (0, 0, 255) if label == "Buồn" else (0, 255, 0)  # Màu đỏ cho "Buồn", xanh lá cho "Vui"
            thickness = 2

            # Vẽ text lên hình ảnh
            cv2.putText(img, text, position, font, font_scale, color, thickness)

            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Hiển thị hình ảnh trong Streamlit
            st.image(img_rgb, caption=filename, use_column_width=True)

# Streamlit UI
st.set_page_config(page_title="Emotion Detection", page_icon="🖼️", layout="wide")  # Cài đặt cấu hình trang

# Tiêu đề chính và mô tả
st.title("Emotion Detection from Images")
st.markdown("""
    This web app allows you to upload an image, and the model will predict whether the image shows **Vui** (Happy) or **Buồn** (Sad).
    Simply upload an image and the app will display the predicted emotion with a confidence score.
    """)
    
# Thêm một chút về mô hình và cách hoạt động của nó
st.markdown("""
    ### How does it work?
    This model was trained using a deep learning algorithm to detect emotions based on facial expressions in images.
    The model is capable of distinguishing between **happy** and **sad** emotions with high accuracy.
""")
# Upload ảnh từ người dùng
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Đọc ảnh
    img = Image.open(uploaded_file)
    
    
    
     # Hiển thị ảnh với chiều rộng 200px và căn giữa ảnh
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{img_to_base64(img)}" width="200">
        </div>
        """, 
        unsafe_allow_html=True
    )
    # Chuyển ảnh thành mảng NumPy
    img_array = np.array(img)
    
    # Chuẩn bị ảnh cho mô hình
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = img_array / 255.0   
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán cảm xúc
    prediction = model.predict(img_array)
    label = "Vui" if prediction[0] < 0.5 else "Buồn"
    confidence = (1 - prediction[0]) * 100 if label == "Vui" else prediction[0] * 100
    confidence_value = confidence.item() if isinstance(confidence, np.ndarray) else confidence  # Tránh lỗi khi confidence là NumPy array

    # Hiển thị kết quả dự đoán
    st.markdown(f"### Prediction: {label}")
    st.markdown(f"**Confidence**: {confidence_value:.2f}%")
    
    # Đưa ra màu sắc tùy thuộc vào kết quả
    if label == "Vui":
        st.markdown('<span style="color:green;">Happy Emotion</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:red;">Sad Emotion</span>', unsafe_allow_html=True)