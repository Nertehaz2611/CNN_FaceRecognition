# USAGE
# python liveness_demo.py

# import the necessary packages
from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"

# Cai dat cac tham so dau vao

# fake_real_128x128.keras
#fake_real_model_omironia_64x64.keras
#miai_80x80.keras
#imironica_48x48.keras
ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", type=str, default='fake_real_v.1.1.pro.keras',
# 	help="path to trained model")
# ap.add_argument("-l", "--le", type=str, default='le.pickle',
# 	help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, default='face_detector',
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load model nhan dien khuon mat
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load model nhan dien fake/real
# print("[INFO] loading liveness detector...")
# model = load_model(args["model"])
# # le = pickle.loads(open(args["le"], "rb").read())
# le = ['fake','real']
# #  Doc video tu webcam
# print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)

time.sleep(2.0)

skip=15
fake_counts =0
real_counts =0
saved = 0
# video_th = 2
a=1
while True:
	# Doc anh tu webcam
	ret,frame = vs.read()

	# Chuyen thanh blob
	(h, w) = frame.shape[:2]
	a+=1
	if a%skip!=0:continue
	a=1
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# Phat hien khuon mat
	net.setInput(blob)
	detections = net.forward()

	# Loop qua cac khuon mat
	
	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		# Neu conf lon hon threshold
		if confidence > args["confidence"]:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			

			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)
			face = frame[startY:endY, startX:endX]
			if face.size==0:continue
			saved+=1
			
			p=os.path.sep.join(["data","raw","train","rin","{}_rin.png".format(saved)])
			# if saved%4==0:
			# 	p = os.path.sep.join(["take_image_val","real_1","{}_{}hieu.png".format(saved,video_th)])
			cv2.imwrite(p,face)
			cv2.rectangle(frame, (startX, startY), (endX, endY),(0,  255,0), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if key==ord("r"):
	# 	fake_counts =0
	# 	real_counts =0
    
	if key == ord("q"):

		break
	if saved>200:
		break

# do a bit of cleanup

vs.release()
cv2.destroyAllWindows()