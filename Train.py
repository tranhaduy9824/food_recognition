import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

dataset_dir = r'F:\Workspace\food_recognition\Training'

# Thiết lập ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    rotation_range=40,
    zoom_range=0.2
)

# Tạo generator cho tập huấn luyện
train_generator = datagen.flow_from_directory(
    directory=dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Tạo generator cho tập kiểm tra
validation_generator = datagen.flow_from_directory(
    directory=dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Lấy số lượng lớp
num_classes = train_generator.num_classes

# Xây dựng mô hình với dropout cho  regularization
def build_model(dropout_rate=0.5):
    model = models.Sequential([
        layers.Input(shape=(150, 150, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    return model

# Hàm để tối ưu hóa và huấn luyện mô hình
def optimize_and_train(max_epochs=20, target_mse=0.2, target_acc=0.7):
    best_mse = float('inf')
    best_acc = 0.0
    best_model = None

    # Thử nghiệm với các giá trị khác nhau của dropout
    for dropout_rate in [0.3, 0.5, 0.7]:
        print(f"Training with parameters: {{'dropout_rate': {dropout_rate}}}")
        model = build_model(dropout_rate)

        # Thiết lập các callback cho early stopping, learning rate reduction, và model checkpoint
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
            ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
        ]

        # Huấn luyện mô hình
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            epochs=max_epochs,
            callbacks=callbacks
        )

        # Đánh giá trên tập validation
        val_loss, val_acc = model.evaluate(validation_generator)
        val_predictions = model.predict(validation_generator)
        val_predictions = np.argmax(val_predictions, axis=-1)

        # Sử dụng đúng nhãn trong validation generator
        true_labels = validation_generator.classes
        val_mse = tf.keras.losses.MeanSquaredError()(tf.keras.utils.to_categorical(true_labels, num_classes),
                                                     tf.keras.utils.to_categorical(val_predictions, num_classes)).numpy()

        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation MSE: {val_mse:.4f}')

        # Cập nhật mô hình tốt nhất nếu tìm thấy
        if val_mse < best_mse and val_acc > best_acc:
            best_mse = val_mse
            best_acc = val_acc
            best_model = model

        # Nếu đạt điều kiện mong muốn thì dừng
        if val_mse <= target_mse and val_acc >= target_acc:
            print(f"Target reached: MSE={val_mse:.4f}, Accuracy={val_acc:.4f}")
            break

    # Lưu mô hình tốt nhất
    if best_model:
        best_model.save('final_modeldk.h5')
    else:
        print("No model met the target criteria.")
    return best_model, best_mse, best_acc

# Gọi hàm tối ưu hóa và huấn luyện
best_model, best_mse, best_acc = optimize_and_train()

# Hàm để vẽ lịch sử huấn luyện với các biểu đồ
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Vẽ biểu đồ độ chính xác
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    # Vẽ biểu đồ loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

# Kiểm tra nếu mô hình tốt nhất tồn tại và vẽ biểu đồ
if best_model:
    plot_training_history(best_model.history)
else:
    print("No model met the target criteria.")

