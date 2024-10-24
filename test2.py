import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys
import io

# Thiết lập mã hóa đầu ra cho console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Tải mô hình đã huấn luyện
model_path = 'final_modeldk.h5'
model = load_model(model_path)

# Kiểm tra số lớp trong mô hình
num_classes = model.output_shape[-1]
print(f"Số lớp trong mô hình: {num_classes}")

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

# Nhãn lớp
class_labels = [
    'Banh beo', 'Banh bot loc', 'Banh can', 'Banh canh', 'Banh chung', 'Banh cuon', 'Banh duc', 
    'Banh gio', 'Banh khot', 'Banh mi', 'Banh pia', 'Banh tet', 'Banh trang nuong', 'Banh xeo', 
    'Bun bo Hue', 'Bun dau mam tom', 'Bun mam', 'Bun rieu', 'Bun thit nuong', 'Ca kho to',
    'Canh chua', 'Cau lau', 'Chao long', 'Com tam', 'Goi cuon', 'Hu tieu', 'My quang', 
    'Nem chua', 'Pho', 'Xoi xeo'
]

# Hàm chọn ảnh từ thư mục
def choose_image():
    img_path = filedialog.askopenfilename(title="Chọn tệp hình ảnh", 
                                           filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if img_path:
        img_array = preprocess_image(img_path)
        predicted_label, prediction = predict_image(model, img_array, class_labels)

        # Hiển thị kết quả dự đoán
        print(f"Xác suất dự đoán: {prediction}")
        print(f"Nhãn dự đoán: {predicted_label}")

        # Hiển thị ảnh và kết quả dự đoán
        img = image.load_img(img_path)
        plt.imshow(img)
        plt.title(f"Dự đoán: {predicted_label}")
        plt.axis('off')  # Ẩn trục
        plt.show()

# Tạo giao diện chính
root = tk.Tk()
root.title("Dự đoán thực phẩm")
root.geometry("400x200")

# Nút chọn ảnh
btn_choose = tk.Button(root, text="Chọn ảnh để dự đoán", command=choose_image)
btn_choose.pack(pady=20)

# Khởi động giao diện
root.mainloop()
