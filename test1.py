import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Tải mô hình đã huấn luyện
model_path = 'final_modeldk.h5'
model = load_model(model_path)

# Đường dẫn tới ảnh cần kiểm tra
img_path = r'F:\Workspace\food_recognition\Test\banhxeooo.jpg'

# Kiểm tra số lớp trong mô hình
num_classes = model.output_shape[-1]
print(f"Number of classes in the model: {num_classes}")

# Tiền xử lý ảnh
def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)  # Thay đổi kích thước
    img_array = image.img_to_array(img)  # Chuyển ảnh thành mảng numpy
    img_array = np.expand_dims(img_array, axis=0)  # Thêm trục batch (batch_size = 1)
    img_array /= 255.0  # Chuẩn hóa giá trị pixel
    return img_array

# Hàm dự đoán và hiển thị kết quả
def predict_image(model, img_array, class_labels):
    prediction = model.predict(img_array)  # Dự đoán nhãn
    predicted_class = np.argmax(prediction, axis=1)  # Lấy chỉ số lớp dự đoán
    predicted_label = class_labels[predicted_class[0]]  # Lấy tên lớp dự đoán
    return predicted_label, prediction


class_labels = [
    'Banh beo', 'Banh bot loc', 'Banh can', 'Banh canh', 'Banh chung', 'Banh cuon', 'Banh duc', 'Banh gio', 'Banh khot', 'Banh mi',
    'Banh pia', 'Banh tet', 'Banh trang nuong', 'Banh xeo', 'Bun bo Hue', 'Bun dau mam tom', 'Bun mam', 'Bun rieu', 'Bun thit nuong', 'Ca kho to',
    'Canh chua', 'Cau lau', 'Chao long', 'Com tam', 'Goi cuon', 'Hu tieu', 'My quang', 'Nem chua', 'Pho', 'Xoi xeo'
]

# Tiền xử lý ảnh
img_array = preprocess_image(img_path)

# Dự đoán nhãn cho ảnh
predicted_label, prediction = predict_image(model, img_array, class_labels)

# Hiển thị kết quả dự đoán
print(f"Prediction probabilities: {prediction}")
print(f"Predicted label: {predicted_label}")

# Hiển thị ảnh và kết quả dự đoán
img = image.load_img(img_path)
plt.imshow(img)
plt.title(f"Predicted: {predicted_label}")
plt.show()