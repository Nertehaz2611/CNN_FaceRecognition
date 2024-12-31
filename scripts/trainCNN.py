import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Khởi tạo các tham số
input_shape = (128, 128, 3)  # Thay đổi theo kích thước ảnh của bạn
batch_size = 32
epochs = 20

# Khởi tạo Data Generator cho dữ liệu huấn luyện và kiểm thử
train_datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    'D:/K22_DUT/HK5/PBL4/CNN/data/processed/train',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'D:/K22_DUT/HK5/PBL4/CNN/data/processed/val',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical'
)

# Xây dựng mô hình CNN
num_classes = train_generator.num_classes  # Đếm số lớp từ generator

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Sử dụng num_classes từ dữ liệu
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Lưu mô hình đã huấn luyện
model.save('D:/K22_DUT/HK5/PBL4/CNN/model.h5')
