from tensorflow.keras.models import load_model
from data_preprocessing import load_and_preprocess_image
import cv2
import numpy as np

# Load mô hình
model = load_model("denoising_autoencoder.h5")

# Load ảnh mới
new_img_path = "data/test/PNEUMONIA/person1_virus_6.jpeg"  # Thay bằng đường dẫn ảnh của bạn
new_img = load_and_preprocess_image(new_img_path)
new_img_input = np.expand_dims(new_img, axis=0)

# Dự đoán
denoised_img = model.predict(new_img_input)[0]

# Lưu ảnh cải thiện
cv2.imwrite("denoised_image.jpg", (denoised_img * 255).astype(np.uint8))

# Hiển thị kết quả
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.imshow(new_img, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(denoised_img, cmap='gray')
plt.title('Denoised Image')
plt.axis('off')
plt.show()