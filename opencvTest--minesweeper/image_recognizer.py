import cv2
import numpy as np
import os
import mss
import time
thresh_IoU=0.35
threshold=0.58
threshold_block=0.6
thresh_IoU_block=0.1
def preprogress_block(img):
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    img_blur=cv2.GaussianBlur(img_gray,(3,3),0)
    edges = cv2.Canny(img_blur, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges
def preprogress(img):
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    img_blur=cv2.GaussianBlur(img_gray,(3,3),0)
    ret,img_binary=cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return img_binary
def match(template,processed_img,img):
    w = template.shape[1]
    h = template.shape[0]
    result=cv2.matchTemplate(processed_img,template,cv2.TM_CCOEFF_NORMED)

    loc=np.where(result>threshold)
    boxes=np.column_stack([loc[1],loc[0],loc[1]+w,loc[0]+h])
    scores=result[loc]
    max_boxes=[]
    while(boxes.size>0):
        index = np.argmax(scores)
        max_boxes.append(boxes[index])
        scores = np.delete(scores, index)
        boxes = np.delete(boxes, index, axis=0)
        indexs=[]
        for index,box in enumerate(boxes,start=0):
            iou=IoU(max_boxes[-1],box)
            if iou > thresh_IoU:
                indexs.append(index)
        mask = np.ones(len(boxes), dtype=bool)
        mask[indexs] = False
        boxes = boxes[mask]
        scores = scores[mask]
    img_copy=img.copy()
    for index,box in enumerate(max_boxes,start=0):
        cv2.rectangle(img_copy,(box[0],box[1]),(box[2],box[3]),(0,0,255),2)
    return img_copy
def IoU(box1,box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    intersection = max(0, inter_x2 - inter_x1) *max(0, inter_y2 - inter_y1)
    box_area1=(x2-x1)*(y2-y1)
    box_area2=(x4-x3)*(y4-y3)
    union=box_area1+box_area2-intersection
    IoU=intersection/union
    return IoU

base_dir = os.path.dirname(os.path.abspath(__file__))
template_folder = os.path.join(base_dir, "temp")
template_num1=cv2.imread(os.path.join(base_dir, "temp","num1.png"))
template_num2=cv2.imread(os.path.join(base_dir, "temp","num2.png"))
template_num3=cv2.imread(os.path.join(base_dir, "temp","num3.png"))
template_num4=cv2.imread(os.path.join(base_dir, "temp","num4.png"))
template_exploded_mine=cv2.imread(os.path.join(base_dir, "temp","exploded_mine.png"))
template_block=cv2.imread(os.path.join(base_dir, "temp","block.png"))
template_mine=cv2.imread(os.path.join(base_dir, "temp","mine.png"))


with mss.mss() as sct:
    monitor = {"top": 284, "left": 280, "width": 1910, "height": 1313}
    time.sleep(3)
    img=np.array(sct.grab(monitor))
    img_copy=img.copy()
    processed_img=preprogress(img_copy)
    processed_template_num1=preprogress(template_num1)
    result=match(processed_template_num1,processed_img,img)
    cv2.imshow("a",result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
