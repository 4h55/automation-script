import cv2
import numpy as np
import os
import mss
import time
from typing import cast
import pyautogui
import read_config
cfg = read_config.read_config()
def get_path():
    templates = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_folder = os.path.join(base_dir, cfg.get('templatePath.folder'))
    template_folder = cast(str, template_folder)  # 断言，无意义
    template_list = cfg.get('templatePath.files')
    for template in template_list:
        template_path = os.path.join(template_folder, template['file'])
        img = cv2.imread(template_path)
        templates[template['name']] = img
    return templates
class ImagePreprocessor:
    def __init__(self,big_img,type):
        self.big_img = big_img
        self.templates = get_path()
        self.type = type
    def preprocess_block(self,img):
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
        img_blur=cv2.GaussianBlur(img_gray,(3,3),0)
        edges = cv2.Canny(img_blur, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.dilate(edges, kernel, iterations=1)
        return edges
    def preprocess_opened_block(self,img):
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_gray)
        img_normalized = cv2.normalize(img_enhanced, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_blur = cv2.GaussianBlur(img_normalized, (3, 3), 0)
        return img_blur
    def preprocess_numAndMine(self,img):
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
        img_blur=cv2.GaussianBlur(img_gray,(3,3),0)
        ret,img_binary=cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return img_binary

    def preprocess(self):#应传入:原图，模板与模板类型
        if self.type == "block":
            processed_img = self.preprocess_block(self.big_img)
            processed_template = self.preprocess_block(self.templates[self.type])
        elif self.type == "num1" or self.type == "num2" or self.type == "num3" or self.type == "num4" or self.type == "mine":
            processed_img = self.preprocess_numAndMine(self.big_img)
            processed_template = self.preprocess_numAndMine(self.templates[self.type])
        elif self.type == "opened_block":
            processed_img = self.preprocess_opened_block(self.big_img)
            processed_template = self.preprocess_opened_block(self.templates[self.type])
        else:
            raise ValueError("there is no such type")
        return processed_img, processed_template,self.type
#应传入:处理过的原图，模板与模板类型.最终返回模板位置
class Deal:
    def __init__(self,preprocess,big_img):
        self.preprocess = preprocess
        self.big_img = big_img
        self.img,self.template,self.type = preprocess.preprocess()
    def match(self):
        w = self.template.shape[1]
        h = self.template.shape[0]
        result=cv2.matchTemplate(self.img,self.template,cv2.TM_CCOEFF_NORMED)
        if self.type in ["num1", "num2", "num3", "num4", "mine"]:
            current_type = "numAndMine"
        else:
            current_type = self.type
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
        for box in max_boxes:
            cv2.rectangle(self.big_img,(box[0],box[1]),(box[2],box[3]),(0,0,255),2)
        cv2.imshow("a", self.big_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return np.array(max_boxes)

    def IoU(self,box1,box2):
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

    def get_position(self):
        boxes=self.match()
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
class State:
    def __init__(self,positions):
        self.positions = positions
    def state_init(self):
        grid = []
        avg_x = []
        avg_y = []
        current_row = [self.positions[0].tolist()]
        for position in self.positions[1:]:
            if abs(position[1] - current_row[0][1]) > 20:
                grid.append(current_row)
                current_row = [position.tolist()]
            else:
                current_row.append(position.tolist())
        grid.append(current_row)

        max_row_length = max(len(row) for row in grid)
        placeholder = [np.nan, np.nan]
        for row in grid:
            shortage = max_row_length - len(row)
            if shortage > 0:
                row += [placeholder] * shortage

        for row in grid:
            y = [position[1] for position in row if not np.isnan(position[1])]
            avg_y.append(int(np.mean(y)))
        col = len(grid[0])
        for col_index in range(col):
            x = [grid[row_index][col_index][0] for row_index in range(len(grid)) if
                 not np.isnan(grid[row_index][col_index][0])]
            avg_x.append(int(np.mean(x)))
        new_y = len(avg_x) + 1
        new_x = len(avg_y) + 1
        new_matrix = np.zeros((new_x, new_y), dtype=object)
        new_matrix[1:, 1:] = -1
        new_matrix[1:, 0] = avg_y
        new_matrix[0, 1:] = avg_x
        new_matrix[0, 0] = "Y\\X"
        return new_matrix
with mss.mss() as sct:
    templates=get_path()
    monitor = {"top": cfg.get('monitor.top'), "left": cfg.get('monitor.left'), "width": cfg.get('monitor.width'), "height": cfg.get('monitor.height')}
    time.sleep(3)
    img=np.array(sct.grab(monitor))
    image_preprocessor=ImagePreprocessor(img,"num1")
    deal=Deal(image_preprocessor,img)
    positions=deal.get_position()
    '''
    state=State(positions)
    new_matrix=state.state_init()
    print(new_matrix)'''


