import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Tải mô hình đã huấn luyện
model = load_model('face_recognition_cnn_model.h5')

# Đường dẫn đến ảnh cần kiểm tra
image_path = "D:/K22_DUT/HK5/PBL4/CNN/data/processed/train/hieu/hieu_4_face0.jpg"  # Sử dụng dấu r để tránh lỗi escape

# Kiểm tra xem file ảnh có tồn tại không
if not os.path.exists(image_path):
    print(f"File ảnh không tồn tại: {image_path}")
else:
    # Đọc ảnh và chuyển đổi sang định dạng phù hợp
    image = cv2.imread(image_path)
    
    # Kiểm tra xem ảnh có được tải thành công không
    if image is None:
        print(f"Không thể tải ảnh từ đường dẫn: {image_path}. Vui lòng kiểm tra đường dẫn và định dạng ảnh.")
    else:
        # Thay đổi kích thước ảnh
        image = cv2.resize(image, (128, 128))  # Thay đổi kích thước ảnh
        image = img_to_array(image) / 255.0  # Chia cho 255 để chuẩn hóa
        image = np.expand_dims(image, axis=0)  # Thêm chiều batch

        # Dự đoán
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)

        # Lấy tên lớp từ chỉ số
        class_labels = ['hcon', 'hieu', 'khoi', 'rin']  # Thay đổi tên lớp cho phù hợp với dữ liệu của bạn
        predicted_label = class_labels[predicted_class[0]]

        print(f'Khuôn mặt được nhận diện là: {predicted_label}')
