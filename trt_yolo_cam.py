import os
import time
import argparse

import cv2
import numpy as np
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from datetime import datetime
import boto3

def parse_args():
	"""Parse input arguments"""
	desc = ('Capture and display live camera video using Jetson Nano, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
	parser = argparse.ArgumentParser(description=desc)
	
	parser.add_argument(
		'-s', '--show_window', type=int, default=0,
		help='[0|1] to display output window. Default is 0.')
	parser.add_argument(
		'-t', '--transmit', type=int, default=0,
		help='[0|1] to send images and data to aws s3 & dynamoDB as set in '
		'awscli. It needs to be set before running.  Default is 0.')
	parser.add_argument(
		'-f', '--flip_val', type=int, default=0,
		help='[2|0|1|3] camera  flip value. Usually 2 or 0. flip_val=0 for camera towards '
		'Jetson, and flip_val=2 for camera outwards Jetson.  Default is 0.')
	parser.add_argument(
		'-c', '--capture_size', type=int, default=608,
		help='[416|608...] camera capture size. default is 608. You should set this value to your yolo input size.')
		
	# Set your wanted default model name here
	model_default='yolov4-tiny-hwan_608'
	parser.add_argument(
		'-m', '--model', type=str, default=model_default,
		help='put your TensorRT yolo model name here. Current defualt is \"'+model_default+'\"')
		
	return parser.parse_args()

def gstreamer_pipeline(
	capture_width=416,
	capture_height=416,
	framerate=30,
	flip_method=2,
	display_width=416,
	display_height=416,
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


def upload_s3(img):
	s3 = boto3.client('s3')
					
	now = datetime.now()
	#change to your file name and saving output directory
	file_name="__your_file_name__"
	file_dir="/home/nano/project/darknet/%s.jpg"%(file_name)
	cv2.imwrite(file_dir,img)
	
	# change to your bucket name
	bucket_name="__your_bucket_name__"
	s3.upload_file(file_dir,bucket_name, '%s'%(file_name),"ContentType": 'image/jpeg'})
	
	s3_url = "https://"+bucket_name+".s3.ap-northeast-2.amazonaws.com/"+file_name
	print("Saved at: "+s3_url)
	
	dynamodb = boto3.resource('dynamodb')
	#change to your table name
	dynamoTable = dynamodb.Table('__your_table_name__')
	dynamoTable.put_item(
	Item={
		'image' : s3_url,
		'time' : '%d-%02d-%02d'%(now.year,now.month,now.day),
		'serial' : '화전',
		'times' : '%d:%02d:%02d'%(now.hour,now.minute,now.second)
		})
		
def detect(cam, trt_yolo, conf_th, vis):
	
	full_scrn = False
	fps = 0.0
	tic = time.time()
	save_time = time.time()
	while True:
		if args.show_window:
			if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
				break
		ret, img = cam.read()
		if img is None:
			break
		boxes, confs, clss = trt_yolo.detect(img, conf_th)
		img = vis.draw_bboxes(img, boxes, confs, clss)
		img = show_fps(img, fps)
		if(args.show_window):
			cv2.imshow(WINDOW_NAME, img)
		toc = time.time()
		curr_fps = 1.0 / (toc - tic)
		# calculate an exponentially decaying average of fps number
		fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
		tic = toc
		
		if(np.count_nonzero(clss == 0) > 0 and time.time()-save_time>=3):
			print("detected no helmet")
			if(args.transmit):
				print("saving image to s3")
				upload_s3(img)
			save_time=time.time()
		key = cv2.waitKey(1)
		if key == 27:  # ESC key: quit program
			break
		elif ((key == ord('F') or key == ord('f')) and args.show_window):  # Toggle fullscreen
			full_scrn = not full_scrn
			set_display(WINDOW_NAME, full_scrn)
			
	cv2.destroyAllWindows()
	cam.release()
        
        ##########################
		
		

def showimg(img):
	cv2.imshow("Image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	#quit()

# get args from cmd line
args = parse_args()
jnID='A'

model_dir = ""

WINDOW_NAME = 'TrtYOLODemo'

cls_dict = get_cls_dict(1)
print(cls_dict)
vis = BBoxVisualization(cls_dict)
#trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)
trt_yolo = TrtYOLO(args.model, 1, False)

#gspipe = ""
gpip = gstreamer_pipeline(
	capture_width=args.capture_size,
	capture_height=args.capture_size,
	flip_method=args.flip_val,
	display_width=args.capture_size,
	display_height=args.capture_size)

cap = cv2.VideoCapture(gpip, cv2.CAP_GSTREAMER)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_size)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_size)
if not cap.isOpened():
	raise IOError("We cannot open Camera")

if(args.show_window):
	open_window(
		    WINDOW_NAME, 'Camera TensorRT YOLO Demo',
		    args.capture_size, args.capture_size)
		    
		    
detect(cap, trt_yolo,conf_th=0.5,vis=vis)

cv2.destroyAllWindows()
cap.release()
	
	
	
	
	
	
	
	
