import os
import random
import shutil

def move_random_images(source_folder, destination_folder, num_images=10):
    # Kiểm tra xem thư mục đích đã tồn tại chưa, nếu chưa thì tạo mới
    os.makedirs(destination_folder, exist_ok=True)
    
    # Lấy danh sách tất cả các file trong thư mục nguồn
    all_images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # Chọn ngẫu nhiên num_images file
    selected_images = random.sample(all_images, min(num_images, len(all_images)))
    
    # Di chuyển từng file sang thư mục đích
    for image in selected_images:
        source_path = os.path.join(source_folder, image)
        destination_path = os.path.join(destination_folder, image)
        shutil.move(source_path, destination_path)

# Thực thi
id = 'rin'
source_folder = f'D:\\K22_DUT\HK5\PBL4\CNN\\data\\processed\\train\\{id}'
destination_folder = f'D:\\K22_DUT\HK5\PBL4\CNN\\data\\processed\\val\\{id}'
move_random_images(source_folder, destination_folder, num_images=20)
