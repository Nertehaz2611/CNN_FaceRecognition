from keras.models import load_model
import cv2
import numpy as np

model = load_model('D:\\K22_DUT\\HK5\\PBL4\\CNN\\model.h5')
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

classes = ['hcon', 'hieu', 'khoi', 'rin', 'other']

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # Khởi tạo biến status và accuracy với giá trị mặc định
    status = 'no face'
    accuracy = 0.0

    if len(faces) != 0:
        for (x, y, w, h) in faces:
            roi_color = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

            # Chuẩn bị ảnh đầu vào
            face_roi = cv2.resize(roi_color, (150, 150))
            X_input = np.array(face_roi).reshape(-1, 150, 150, 3).astype('float32') / 255

            # Dự đoán
            prediction = model.predict(X_input)
            Predict = np.argmax(prediction, axis=-1)
            status = classes[int(Predict)]
            accuracy = np.max(prediction) * 100  # Độ chính xác của dự đoán

    # Hiển thị nhãn và độ chính xác
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"{status} - {accuracy:.2f}%", (x,max(0,y-20)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Drowsiness Detection Tutorial', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
