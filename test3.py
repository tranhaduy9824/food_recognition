import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
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
def preprocess_image(img_array, target_size=(150, 150)):
    img_array = cv2.resize(img_array, target_size)  # Thay đổi kích thước
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # Đổi màu BGR sang RGB
    img_array = np.expand_dims(img_array, axis=0)  # Thêm trục batch
    img_array = img_array.astype('float32') / 255.0  # Chuẩn hóa
    return img_array

# Hàm dự đoán và hiển thị kết quả
def predict_image(model, img_array, class_labels):
    prediction = model.predict(img_array)  # Dự đoán nhãn
    predicted_class = np.argmax(prediction, axis=1)  # Lấy chỉ số lớp dự đoán
    predicted_label = class_labels[predicted_class[0]]  # Lấy tên lớp dự đoán
    return predicted_label, prediction

# Nhãn lớp
class_labels = [
    'Banh beo', 'Banh bot loc', 'Banh can', 'Banh canh', 'Banh chung', 'Banh cuon', 
    'Banh duc', 'Banh gio', 'Banh khot', 'Banh mi', 'Banh pia', 'Banh tet', 
    'Banh trang nuong', 'Banh xeo', 'Bun bo Hue', 'Bun dau mam tom', 
    'Bun mam', 'Bun rieu', 'Bun thit nuong', 'Ca kho to',
    'Canh chua', 'Cau lau', 'Chao long', 'Com tam', 
    'Goi cuon', 'Hu tieu', 'My quang', 'Nem chua', 'Pho', 'Xoi xeo'
]

# Hàm chọn ảnh từ thư mục
def choose_image():
    img_path = filedialog.askopenfilename(title="Chọn tệp hình ảnh", 
                                           filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if img_path:
        img = image.load_img(img_path)
        img_array = image.img_to_array(img)
        img_array = preprocess_image(img_array)
        
        predicted_label, prediction = predict_image(model, img_array, class_labels)

        # Hiển thị kết quả dự đoán
        print(f"Xác suất dự đoán: {prediction}")
        print(f"Nhãn dự đoán: {predicted_label}")

        # Hiển thị ảnh và kết quả dự đoán
        plt.imshow(img)
        plt.title(f"Dự đoán: {predicted_label}")
        plt.axis('off')  # Ẩn trục
        plt.show()

# Hàm nhận diện từ camera
def detect_from_camera():
    cap = cv2.VideoCapture(0)  # Thay đổi 0 thành 1 nếu camera ngoài là camera thứ hai

    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Lỗi", "Không thể đọc khung từ camera.")
            break

        # Tiền xử lý và dự đoán cho mỗi khung hình
        img_array = preprocess_image(frame)
        predicted_label, prediction = predict_image(model, img_array, class_labels)

        # Hiển thị kết quả
        print(f"Xác suất dự đoán: {prediction}")
        print(f"Nhãn dự đoán: {predicted_label}")

        # Vẽ nhãn dự đoán lên khung hình
        cv2.putText(frame, f"Dự đoán: {predicted_label}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Hiển thị video
        cv2.imshow('Camera', frame)

        # Nhấn 'q' để thoát
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Tạo giao diện chính
root = tk.Tk()
root.title("Dự đoán thực phẩm")
root.geometry("400x200")

# Nút chọn ảnh
btn_choose = tk.Button(root, text="Chọn ảnh để dự đoán", command=choose_image)
btn_choose.pack(pady=20)

# Nút nhận diện từ camera
btn_camera = tk.Button(root, text="Nhận diện từ camera", command=detect_from_camera)
btn_camera.pack(pady=20)

# Khởi động giao diện
root.mainloop()
