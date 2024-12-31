import os
import cv2

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # Thay đổi kích thước ảnh
    image = cv2.resize(image, (224, 224))
    # Chuyển đổi sang định dạng RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def process_data(data_dir):
    # Tạo thư mục processed nếu chưa tồn tại
    processed_dir = "D:/K22_DUT/HK5/PBL4/CNN/data/processed/train"
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    for person in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person)
        processed_person_path = os.path.join(processed_dir, person)

        if not os.path.exists(processed_person_path):
            os.makedirs(processed_person_path)

        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            processed_img = preprocess_image(img_path)

            # Lưu ảnh đã tiền xử lý
            processed_img_filename = f"{os.path.splitext(img)[0]}.jpg"
            cv2.imwrite(os.path.join(processed_person_path, processed_img_filename), cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    raw_data_dir = "D:/K22_DUT/HK5/PBL4/CNN/data/raw/train"
    process_data(raw_data_dir)
    print("Done")
