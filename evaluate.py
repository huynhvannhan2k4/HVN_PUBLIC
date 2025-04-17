from tensorflow.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
import numpy as np
from data_preprocessing import load_and_preprocess_image

# Load mô hình
model = load_model("denoising_autoencoder.h5")

# Load ảnh test mẫu
test_img_path = "data/test/NORMAL/IM-0001-0001.jpeg"
test_img = load_and_preprocess_image(test_img_path)
test_img_input = np.expand_dims(test_img, axis=0)  # Thêm batch dimension

# Dự đoán
predicted_img = model.predict(test_img_input)[0]

# Tính PSNR và SSIM
psnr = peak_signal_noise_ratio(test_img, predicted_img)
ssim = structural_similarity(test_img, predicted_img, multichannel=False)

print(f"PSNR: {psnr:.2f}")
print(f"SSIM: {ssim:.2f}")

# Visualize
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.imshow(test_img, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(predicted_img, cmap='gray')
plt.title('Denoised Image')
plt.axis('off')
plt.savefig("comparison.png")
plt.show()