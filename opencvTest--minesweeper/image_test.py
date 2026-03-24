import cv2
import numpy as np
import os
import mss
import time
from typing import cast
import read_config
import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1
cfg = read_config.read_config()

def get_path():
    templates = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_folder = os.path.join(base_dir, cfg.get('templatePath.folder'))
    template_folder = cast(str, template_folder)  # 断言，无意义
    template_list = cfg.get('templatePath.files')
    for template in template_list:
        template_path = os.path.join(template_folder, template['file'])
        img = cv2.imread(template_path,cv2.IMREAD_UNCHANGED)
        templates[template['type']] = img
    return templates
templates=get_path()
class ImagePreprocessor:
    def __init__(self,big_img):
        self.big_img = big_img
    def preprocess_block(self,img):
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
        img_blur=cv2.GaussianBlur(img_gray,(3,3),0)
        edges = cv2.Canny(img_blur, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.dilate(edges, kernel, iterations=1)
        return edges
    def preprocess_opened_block(self,img):
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
        img_normalized = cv2.normalize(img_blur, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return img_normalized

    def preprocess_num(self, img, type):
        # 先转HSV颜色空间，按色相精准筛选颜色
        if img.shape[-1] == 4:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            img_bgr = img.copy()
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # ✅ 扫雷数字的HSV颜色范围（精准到色相，彻底过滤其他颜色）
        color_range = {
            "num1": [np.array([100, 120, 70]), np.array([130, 255, 255])],  # 蓝色1
            "num2": [np.array([40, 120, 70]), np.array([77, 255, 255])],  # 绿色2
            "num3": [np.array([0, 120, 70]), np.array([10, 255, 255])],  # 红色3
            "num4": [np.array([100, 120, 70]), np.array([130, 255, 255])],  # 深蓝色4
        }

        # 按类型生成颜色掩码，只保留目标颜色的数字
        if type in color_range:
            lower, upper = color_range[type]
            mask = cv2.inRange(img_hsv, lower, upper)
        else:
            # lose等其他类型用灰度
            mask = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # 后续处理和你的原逻辑完全对齐，保证模板匹配一致
        img_blur = cv2.GaussianBlur(mask, (3, 3), 0)
        ret, img_binary = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_clean = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=1)

        return img_clean

    def preprocess(self,type):
        if type == "block":
            processed_img = self.preprocess_block(self.big_img)
            processed_template = self.preprocess_block(templates[type])
        elif type == "num1" or type == "num2" or type == "num3" or type == "num4"or type == "lose" :
            processed_img = self.preprocess_num(self.big_img,type)
            processed_template = self.preprocess_num(templates[type],type)
        elif type == "opened_block":
            processed_img = self.preprocess_opened_block(self.big_img)
            processed_template = self.preprocess_opened_block(templates[type])
        else:
            raise ValueError("there is no such type")
        return processed_img, processed_template
#应传入:处理过的原图，模板与模板类型.最终返回模板位置
class Deal:
    def __init__(self,big_img):
        self.big_img = big_img
    def match(self,img,template,type):
        w = template.shape[1]
        h = template.shape[0]
        result=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
        if type in ["num1", "num2", "num3", "num4"]:
            current_type = "num"
        else:
            current_type = type
        score_thresh = cfg.get(f"threshParam.thresh_{current_type}")
        iou_thresh = cfg.get(f"threshParam.thresh_IoU_{current_type}")
        loc=np.where(result>score_thresh)
        boxes=np.column_stack([loc[1],loc[0],loc[1]+w,loc[0]+h])
        scores=result[loc]

        boxes_nms = boxes.copy()
        boxes_nms[:, 2] = boxes_nms[:, 2] - boxes_nms[:, 0]  # w = x2 - x1
        boxes_nms[:, 3] = boxes_nms[:, 3] - boxes_nms[:, 1]  # h = y2 - y1
        max_boxes = np.array([])
        if boxes.size > 0:
            indices = cv2.dnn.NMSBoxes(boxes_nms.tolist(),scores.tolist(),score_threshold=score_thresh,nms_threshold=iou_thresh)
            indices = np.array(indices)
            if indices.size > 0:
                indices = indices.ravel()
                max_boxes = boxes[indices]


        if type == "opened_block" and max_boxes.size > 0:
            valid_boxes = []
            for box in max_boxes:
                x1, y1, x2, y2 = box.astype(int)
                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                variance = np.var(roi)
                if variance < 500:
                    valid_boxes.append(box)

            max_boxes = np.array(valid_boxes)


        for box in max_boxes:
            cv2.rectangle(self.big_img,(box[0],box[1]),(box[2],box[3]),(0,0,255),2)
        cv2.imshow("a", self.big_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return np.array(max_boxes)


    def get_position(self,img,template,type):
        boxes=self.match(img,template,type)
        if boxes.size == 0 or len(boxes.shape) == 1:
            return np.array([])
        center_x = ((boxes[:, 0] + boxes[:, 2]) / 2).astype(int)
        center_y = ((boxes[:, 1] + boxes[:, 3]) / 2).astype(int)
        position = np.column_stack([center_x,center_y ])
        if position.size == 0 or len(position.shape) == 1:
            return np.array([])
        index = np.lexsort((position[:, 0], position[:, 1]))
        sorted_position = position[index]
        return sorted_position
class Tool:

    def click(self,x,y):
        pyautogui.click(x,y)
    def screenshot(self):
        with mss.mss() as sct:
            monitor = {"top": cfg.get('monitor.top'), "left": cfg.get('monitor.left'),
                       "width": cfg.get('monitor.width'), "height": cfg.get('monitor.height')}
            time.sleep(3)
            img = np.array(sct.grab(monitor))
        return img
if __name__ == "__main__":
    tool=Tool()
    img = tool.screenshot()
    preprocess = ImagePreprocessor(img)
    deal = Deal(img)
    type=("num1"
          ""
          ""
          ""
          ""
          ""
          ""
          ""
          ""
          ""
          "")
    processed_img, processed_template = preprocess.preprocess(type)
    cv2.imshow(f"1. 预处理后的游戏截图 ({type})", processed_img)
    cv2.imshow(f"2. 预处理后的模板 ({type})", processed_template)
    print(f"模板尺寸: {processed_template.shape}, 截图预处理尺寸: {processed_img.shape}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    positions = deal.get_position(processed_img, processed_template, type)
    print(positions)