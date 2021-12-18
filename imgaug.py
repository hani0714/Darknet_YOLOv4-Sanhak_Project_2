import imgaug.augmenters as iaa
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
import cv2
import os
import re


def convert_yolo_imgaug_label(labeldata, w, h):
    labelno = int(labeldata[0])
    x_center = float(labeldata[1]) * w
    y_center = float(labeldata[2]) * h
    width = float(labeldata[3]) * w
    height = float(labeldata[4]) * h

    _x1 = int(x_center-width/2)
    _y1 = int(y_center-height/2)
    _x2 = int(x_center+width/2)
    _y2 = int(y_center+height/2)
    return BoundingBox(x1=_x1,y1=_y1,x2=_x2,y2=_y2)
    

#해상도에 따른 label 계산
def get_bbox_data(bboxtxt, width=608, height=608):
    bboxlist = []
    for line in bboxtxt:
        if(line==''):
            continue
        bboxdata = line.split(" ")
        if(len(bboxdata)==5):
            bboxlist.append(convert_yolo_imgaug_label(bboxdata, width, height))
    return bboxlist

def load_img_txt_from_folder(folder):
    images=[]
    imgnames=[]
    bbox=[]
    listd=os.listdir(folder)
    listd.sort()
    print(listd)
    for filename in listd:
        name = filename.split('.')
        if(len(name)!=2):
            print('[Warning]nameing convention not followed by :'+filename)
            continue
        if(name[1]!='jpg' and name[1]!='txt'):
            print('[Warning]nameing convention not followed by :'+filename)
            continue
        if(name[1]=='jpg'):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
                imgnames.append(name[0])
        elif(name[1]=='txt'):
            with open(os.path.join(folder,filename),'r',encoding='UTF8') as ftxt:
                txt=ftxt.readlines()
                #calculate label box to imgaug style
                newbbox=get_bbox_data(txt)
                if(newbbox!=None):
                    bbox.append(newbbox)
    return images,bbox,imgnames





def write_images(name, number,images):
    for i in range(len(images)):
        cv2.imwrite('result\\%s_%d.jpg'%(name[i]+number, i), images[i]) #이미지 저장할 경로 설정을 여기서 한다.
        #cv2.imshow('image', images[i])
        #print("image saving complete", i)
    print("image saving complete")
    
def write_img_label(name, images, labels):
    for i in range(len(images)):
        cv2.imwrite('result\\%s_%d.jpg'%(name, i), images[i])
        f = open('result\\%s_%d.txt'%(name, i),'w')
        for lb in labels[i]:
            print(lb)
            f.write('0 ')
            for j in range(0,4):
                if j==3:
                    f.write(str(lb[j])+"\n")
                else:
                    f.write(str(lb[j])+" ")
        f.close()


def to_yolo_label(bbox, width=608, height=608):
    lblist=[]
    bbox=bbox.remove_out_of_image().clip_out_of_image()
    for i in range(len(bbox.bounding_boxes)):
        x1=bbox.bounding_boxes[i].x1
        y1=bbox.bounding_boxes[i].y1
        x2=bbox.bounding_boxes[i].x2
        y2=bbox.bounding_boxes[i].y2
        #print(type(x1),type(y1),type(x2),type(y2))
        lblist.append([((x1+x2)/2)/width, ((y1+y2)/2)/height, (x2-x1)/width, (y2-y1)/height])
    return lblist
    
#이미지 증강 코드
def augmentations3(image,bbox,imgname):
    bbs = BoundingBoxesOnImage(bbox,shape=image.shape)

    #seq2 = iaa.ChannelShuffle(p=1.0)
    #seq3 = iaa.Dropout((0.05, 0.1), per_channel=0.5)

    seq1 = iaa.Fliplr(0.5)
    seq2 = iaa.Sequential([
        iaa.Affine(
        translate_px={"x": (-30,30), "y":(-30,30)},
        scale=(0.8, 0.9),
        rotate=(-15,15)
        ), iaa.Fliplr(0.5)
    ])                
    seq3 = iaa.Sequential([
        iaa.Affine(
        translate_px={"x": (-30,30), "y":(-30,30)},
        scale=(0.8, 0.9),
        rotate=(-15,15)
        ), iaa.Fliplr(0.5)
    ])
    seq4 = iaa.Sequential([
        iaa.Affine(
        translate_px={"x": (-30,30), "y":(-30,30)},
        scale=(0.8, 0.9),
        rotate=(-15,15)
        ), iaa.Fliplr(0.5)
    ])
    seq5 = iaa.Sequential([
        iaa.Affine(
        translate_px={"x": (-30,30), "y":(-30,30)},
        scale=(0.8, 0.9),
        rotate=(-15,15)
        ), iaa.Fliplr(0.5)
    ])
    seq6 = iaa.Sequential([
        iaa.Affine(
        translate_px={"x": (-30,30), "y":(-30,30)},
        scale=(0.8, 0.9),
        rotate=(-15,15)
        ), iaa.Fliplr(0.5)
    ])
    print("image augmentation beginning")
    img1,bbs_aug1=seq1(image=image,bounding_boxes=bbs)
    print("sequence 1 completed......")
    img2,bbs_aug2=seq2(image=image,bounding_boxes=bbs)
    print("sequence 2 completed......")
    img3,bbs_aug3=seq3(image=image,bounding_boxes=bbs)
    print("sequence 3 completed......")
    img4,bbs_aug4=seq4(image=image,bounding_boxes=bbs)
    print("sequence 4 completed......")
    img5,bbs_aug5=seq5(image=image,bounding_boxes=bbs)
    print("sequence 5 completed......")
    img6,bbs_aug6=seq6(image=image,bounding_boxes=bbs)
    print("sequence 6 completed......")
    imglist = [img1, img2, img3, img4, img5, img6]
    lblist = [to_yolo_label(bbs_aug1),to_yolo_label(bbs_aug2),to_yolo_label(bbs_aug3),to_yolo_label(bbs_aug4),to_yolo_label(bbs_aug5),to_yolo_label(bbs_aug6)]
    #lblist = [bbs_aug1.bounding_boxes,bbs_aug2.bounding_boxes,bbs_aug3.bounding_boxes,bbs_aug4.bounding_boxes,bbs_aug5.bounding_boxes,bbs_aug6.bounding_boxes]
    print("writing images...")
    write_img_label(imgname, imglist, lblist)
    print("Done! proceed to next augmentations")
    

photos = 'C:\\Users\\Desktop\\img' #이미지 읽어올 경로
fullList = os.listdir(photos)
fullList.sort()
#texts = 'C:\\Users\\Mster\\Desktop\\DataSet\\augsrc' #이미지 읽어올 경로
#bbox = os.listdir(texts)
print (fullList)

#photos1 = load_images_from_folder(photos)
#photos2 = load_images_from_folder(os.path.join(photos, folders[1]))
#photos3 = load_images_from_folder(os.path.join(photos, folders[2]))

#txts1 = load_txt_from_folder(texts)
images, bboxes, imgnames = load_img_txt_from_folder(photos)
print(imgnames)
print(bboxes)

for i in range(len(imgnames)):
    augmentations3(images[i],bboxes[i],imgnames[i])
