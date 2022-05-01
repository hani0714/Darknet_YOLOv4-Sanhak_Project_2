# Darknet YOLOv4 - Sanhak Project 2
## 멘토기업: 테크노니아

### filename.py
- 이미지파일 이름 변경시 사용

### txtmake.py 
- 빈 txt파일 생성. labelimg 사용시 미리 빈 txt파일 생성하는게 편함

### set-divide.py 
- 이미지 파일을 training set, valid set 으로 random 하게 나눠줌

### imgaug.py 
- 이미지 augmemtation 에 사용

### image_opencv.cpp
- line 1018 ~ 1037
- detection 발생 시 캡쳐 후 이미지 저장 부분 추가

### pyolo.py 
- opencv의 dnn모듈을 사용해 yolo로 single image detection

### pycamyolo.py 
- pyolo에서 jetson nano의 cameara를 사용해 영상 detection
- Jetson Nano 이미지에 기본적으로 설치되어 있는 opencv로 실행할 경우 오류가 나므로 높은 버전의 opencv를 설치해야함


### trt_yolo_cam.py 
- Yolo에서 tensorRT로 변환된 모델로 Camera Detection 실행
- https://github.com/jkjung-avt/tensorrt_demos 에서 trt_yolo.py를 수정했음
