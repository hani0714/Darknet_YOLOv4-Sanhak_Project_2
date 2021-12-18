import cv2
import numpy as np
import time
import sys
import boto3
import argparse
from datetime import datetime

def parser():
	parser = argparse.ArgumentParser(description="YOLO Object Detection")
	parser.add_argument("--show", type=int, default=1, help="video source. If empty, uses webcam 0 stream")
	return parser.parse_args()


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    framerate=30,
    flip_method=0,
    display_width=1280,
    display_height=720,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx !"
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

#AWS ID and key
'''gyu
my_id = "AKIAYYKS5IWSR2RNQEG2"
my_key = "Q8j9wqX8Uw0L22lPZuul2GN0JX4GSXQevxWTNkDe"
bucket_name = 'yolo-detected'
'''

my_id = "AKIA6KQFKZFVOXF5QEDJ"
my_key = "8dRkJw5Hv0+JYq8DWa9H50sQq+r56KLT0JzPG/q3"
bucket_name = 'detected'

def upload_s3(img):
	s3 = boto3.client(
					's3',
					aws_access_key_id=my_id,
					aws_secret_access_key=my_key)
					
	now = datetime.now()
	#bucket name where you want to upload image
	file_name="%s_%d%02d%02d_%02d%02d%02d"%(jnID, now.year, now.month, now.day, now.hour, now.minute, now.second)
	file_dir="/home/hwang/project/darknet/outputimg/%s.jpg"%(file_name)
	cv2.imwrite(file_dir,img)
	
	s3.upload_file(file_dir,bucket_name, '%s/%s'%('test1',file_name), ExtraArgs={'ACL':'public-read',"ContentType": 'image/jpeg'})
	
	s3_url = "https://"+bucket_name+".s3.ap-northeast-2.amazonaws.com/test1/"+file_name
	print("Saved at: "+s3_url)
	
	dynamodb = boto3.resource('dynamodb')
	dynamoTable = dynamodb.Table('detect')
	dynamoTable.put_item(
	Item={
		'image' : s3_url,
		'time' : '%d-%02d-%02d'%(now.year,now.month,now.day),
		'serial' : '화전'
		})

def detect(img, score_thresh, nms_thresh):

	# Check img's shape
	height, width, channels = img.shape

	# Detecting objects
	blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)

	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5:
				# Object detected
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)
				# Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, nms_thresh)
	font = cv2.FONT_HERSHEY_PLAIN
	nohelmet_count=0
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			#if class_ids[i] == 0:
			nohelmet_count = nohelmet_count+1
			color = colors[i]
			cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
			cv2.putText(img, label, (x, y + 30), font, 1, color, 3)
			
	return img, nohelmet_count

def showimg(img):
	cv2.imshow("Image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	#quit()

# get command arguments
args = parser()
print(args.show)
jnID='A'
#For delay in saving
save_time=time.time()

# Load Yolo
#net = cv2.dnn.readNet("custom/yolov4-tiny-gyu_best.weights", "/home/hwang/project/darknet/cfg/yolov4-tiny-custom.cfg")
net = cv2.dnn.readNetFromDarknet("/home/hwang/project/darknet/cfg/yolov4-tiny-custom.cfg", "custom/yolov4-tiny-hwan_13.weights")
#net = cv2.dnn.readNet("yolov4-tiny.weights", "cfg/yolov4-tiny.cfg")

#use CUDA
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = []
with open("data/kb_helmet.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes),3))

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
	raise IOError("We cannot open Camera")

#img = cv2.imread("testimg/5003.jpg")
while True:
	timer=time.time()
	ret, img = cap.read()
	#cv2.resize(img, None, fx=0.4, fy=0.6)
	img, hc = detect(img,0.5,0.4)
	print('[Info] Time Taken: {} | FPS: {} | {}-No Helmet'.format(time.time() - timer, 1/(time.time() - timer),hc), end='\r')
	if(args.show):
		cv2.imshow("Web cam input", img)
	#print('[Info] Time Taken: {} | FPS: {}'.format(time.time() - timer, 1/(time.time() - timer)), end='\r')
	if(hc > 0 and time.time()-save_time>=2):
		upload_s3(img)
		save_time=time.time()
		
	if(cv2.waitKey(25) & 0xFF == ord("q")):
		cap.close()
		cv2.destroyAllWindows()
		break
	
	
	
	
	
	
	
	
	
