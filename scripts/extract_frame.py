import cv2
import os

# Đường dẫn đến video
id = "khoi"
video_path = f'D:\K22_DUT\HK5\PBL4\CNN\\videos\\{id}.mp4'
# Đường dẫn lưu các frame
output_dir = f'D:\\K22_DUT\HK5\PBL4\CNN\\data\\raw\\train\\{id}'

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Mở video
cap = cv2.VideoCapture(video_path)

# Lấy tổng số frame và thời gian video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
video_duration = total_frames / fps  # Tổng thời gian video (giây)

# Tính toán khoảng thời gian giữa mỗi frame
num_frames_to_extract = 200
time_between_frames = video_duration / num_frames_to_extract  # Khoảng thời gian giữa mỗi frame (giây)

frame_count = 0
extracted_count = 0

while cap.isOpened() and extracted_count < num_frames_to_extract:
    ret, frame = cap.read()
    
    if not ret:
        print("Cannot receive frame (stream end?). Exiting ...")
        break

    # Tính thời gian của frame hiện tại
    current_time = frame_count / fps

    # Nếu thời gian của frame này vượt quá thời gian giữa các frame, lưu frame
    if current_time >= extracted_count * time_between_frames:
        # Lưu frame dưới dạng ảnh
        frame_path = os.path.join(output_dir, f'{id}_{extracted_count}.jpg')
        cv2.imwrite(frame_path, frame)
        extracted_count += 1

    frame_count += 1

# Giải phóng bộ nhớ
cap.release()
cv2.destroyAllWindows()

print(f"Extracted {extracted_count} frames from the video.")
