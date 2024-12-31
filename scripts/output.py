import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
import cv2
import numpy as np
import time
import mysql.connector
from datetime import datetime

# Load model
model = load_model("D:/K22_DUT/HK5/PBL4/CNN/checkpoints/checkpoint-7/model-7.h5")
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

classes = ['3', '1', '2', '4', '5']

# Function to predict face class
def predict_face(face_image):
    X_input = np.array(face_image).reshape(-1, 150, 150, 3).astype('float32') / 255
    prediction = model.predict(X_input, verbose=0)
    class_index = np.argmax(prediction, axis=-1)
    confidence = np.max(prediction) * 100
    return classes[int(class_index)], confidence

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",        # Thay đổi nếu cần
    port = 3306,
    username = "root",
    password = "",
    database="pbl4" # Tên cơ sở dữ liệu của bạn
)
cursor = db.cursor()

# Function to insert attendance into the database
def insert_attendance(user_id, timestamp):
    query = "INSERT INTO chamcong (staffID, time) VALUES (%s, %s)"
    cursor.execute(query, (user_id, timestamp))
    db.commit()

def insert_staffAttendance(lastTimeKeeping, staff_id):
    query = "UPDATE nhanvien SET lastTimeKeeping = %s, timeKeepingStatus = 1 WHERE IDNV = %s"
    cursor.execute(query, (lastTimeKeeping, staff_id))
    db.commit()

def checkAttendance(staff_id):
    query = "SELECT timeKeepingStatus FROM nhanvien WHERE IDNV = %s"
    cursor.execute(query, (staff_id,))
    result = cursor.fetchone()
    if result:
        return result[0] 
    else:
        return None

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        # Start the timer for 3 seconds
        start_time = time.time()
        results = []

        while time.time() - start_time < 3:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                roi_color = frame[y:y + h, x:x + w]
                face_roi = cv2.resize(roi_color, (150, 150))

                # Predict
                status, confidence = predict_face(face_roi)
                results.append(confidence)  # Add confidence to the list

                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Display on the frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f"{status} - {confidence:.2f}%", (x, max(0, y-20)), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Face Recognition', frame)

            # Break if 'q' is pressed
            if cv2.waitKey(2) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # Calculate the average confidence over the 3 seconds
        if results:
            avg_confidence = np.mean(results)  # Calculate average confidence

            # Check if the average confidence is greater than 80%
            if avg_confidence > 80:
                print(f"Final Recognition Result: {status} \nAverage Recognition Confidence: {avg_confidence:.2f}% \n\n")
                if status != 'unknown':
                    if not checkAttendance(status):
                        # Get the current timestamp
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        # Insert attendance data into the database
                        user_id = status  # Assuming index matches the user id
                        insert_attendance(user_id, timestamp)
                        insert_staffAttendance(timestamp, user_id)
                    else:
                        #Do nothing
                        print("You have been checked-in.")
            else:
                print("No face recognized with high enough average confidence.")

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
