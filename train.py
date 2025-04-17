from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from model import build_denoising_autoencoder
import os

# Đường dẫn đến dữ liệu
train_dir = "data/train"
val_dir = "data/val"

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

# Load data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=32,
    class_mode='input'  # Dùng cho autoencoder
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=32,
    class_mode='input'
)

# Build và compile model
model = build_denoising_autoencoder()
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Huấn luyện
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator
)

# Lưu mô hình
model.save("denoising_autoencoder.h5")

# Visualize loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("training_loss.png")
plt.show()