import os

# Đường dẫn đến folder chứa ảnh
folder_path = "data\\raw\\train\\hieu"

# Lặp qua tất cả các file trong folder
for filename in os.listdir(folder_path):
    # Kiểm tra nếu tên file chứa 'hieu'
    if '2hieu' in filename:
        # Đổi tên file
        new_filename = filename.replace('2hieu', 'hieu')
        # Lấy đường dẫn đầy đủ của file cũ và mới
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)
        # Đổi tên file
        os.rename(old_file, new_file)
        print(f"Đổi tên {filename} thành {new_filename}")
