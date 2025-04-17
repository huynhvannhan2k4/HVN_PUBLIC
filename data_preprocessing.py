import cv2
import numpy as np
from skimage import exposure

def load_and_preprocess_image(path, size=(256, 256)):
    """Tải và tiền xử lý ảnh X-quang."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh grayscale
    img = cv2.resize(img, size)  # Chuẩn hóa kích thước
    img = exposure.equalize_adapthist(img, clip_limit=0.03)  # Tăng cường độ tương phản bằng CLAHE
    img = img / 255.0  # Chuẩn hóa giá trị pixel về [0,1]
    return img

# Ví dụ sử dụng
if __name__ == "__main__":
    sample_img = load_and_preprocess_image("data/train/NORMAL/IM-0115-0001.jpeg")
    cv2.imwrite("preprocessed_sample.jpg", (sample_img * 255).astype(np.uint8))