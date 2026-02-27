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
        img = cv2.imread(template_path)
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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_gray)
        img_normalized = cv2.normalize(img_enhanced, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_blur = cv2.GaussianBlur(img_normalized, (3, 3), 0)
        return img_blur
    def preprocess_num(self,img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        ret, img_binary = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_clean = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=1)

        return img_clean

    def preprocess(self,type):
        if type == "block":
            processed_img = self.preprocess_block(self.big_img)
            processed_template = self.preprocess_block(templates[type])
        elif type == "num1" or type == "num2" or type == "num3" or type == "num4"or type == "lose" :
            processed_img = self.preprocess_num(self.big_img)
            processed_template = self.preprocess_num(templates[type])
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
        '''
        for box in max_boxes:
            cv2.rectangle(self.big_img,(box[0],box[1]),(box[2],box[3]),(0,0,255),2)
        cv2.imshow("a", self.big_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''
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
class State:
    def __init__(self):
        self.grid=[]

    def state_init(self,big_img,type="block"):
        preprocess=ImagePreprocessor(big_img)
        deal=Deal(big_img)
        processed_img, processed_template = preprocess.preprocess(type)
        positions=deal.get_position(processed_img, processed_template,type)

        avg_x = []
        avg_y = []
        current_row = [positions[0].tolist()]
        for position in positions[1:]:
            if abs(position[1] - current_row[0][1]) > 20:
                self.grid.append(current_row)
                current_row = [position.tolist()]
            else:
                current_row.append(position.tolist())
        self.grid.append(current_row)

        max_row_length = max(len(row) for row in self.grid)
        placeholder = [np.nan, np.nan]
        for row in self.grid:
            shortage = max_row_length - len(row)
            if shortage > 0:
                row += [placeholder] * shortage

        for row in self.grid:
            y = [position[1] for position in row if not np.isnan(position[1])]
            avg_y.append(int(np.mean(y)))
        col = len(self.grid[0])
        for col_index in range(col):
            x = [self.grid[row_index][col_index][0] for row_index in range(len(self.grid)) if
                 not np.isnan(self.grid[row_index][col_index][0])]
            avg_x.append(int(np.mean(x)))
        new_y = len(avg_x) + 1
        new_x = len(avg_y) + 1
        new_matrix = np.zeros((new_x, new_y), dtype=object)
        new_matrix[1:, 1:] = -1
        new_matrix[1:, 0] = avg_y
        new_matrix[0, 1:] = avg_x
        new_matrix[0, 0] = 0#未触发地雷
        self.grid = new_matrix
    def state_refresh(self):
        act=Act()
        act.click(1404,884)
        img=act.screenshot()
        type_to_value = {
            item['type']:item['value']
            for item in cfg.get("templatePath.files")
        }
        preprocess = ImagePreprocessor(img)
        deal=Deal(img)
        x_list=self.grid[0,1:]
        x_path={x:index for index,x in enumerate(x_list,start=1)}
        y_list=self.grid[1:,0]
        y_path={y:index for index,y in enumerate(y_list,start=1)}
        max_loop=1
        loop=0
        while self.grid[0,0]==0 and loop<max_loop :
            '''
            processed_img, processed_template = preprocess.preprocess(type="lose")
            lose_position = deal.get_position(processed_img, processed_template, type="lose")
            if not lose_position:
                self.grid[0, 0] = 1
            else:
                continue'''
            for type,template in list(templates.items()):
                if type in ["block", "lose"]:
                    continue
                value=type_to_value[type]
                processed_img, processed_template=preprocess.preprocess(type)
                positions=deal.get_position(processed_img, processed_template,type)
                print(type)
                print(positions)
                for x, y in positions:
                    col_idx = self._find_closest_key(x_path, int(x), threshold=20)
                    row_idx = self._find_closest_key(y_path, int(y), threshold=20)

                    if col_idx is None or row_idx is None:
                        print(f"坐标 ({x},{y}) 未匹配到网格，跳过")
                        continue
                    self.grid[row_idx, col_idx] = value
                    print(f"更新网格：({row_idx},{col_idx}) = {value}")

            loop += 1

        return self.grid

    def _find_closest_key(self, mapping, target, threshold=20):
        if not mapping:
            return None
        closest_key = min(mapping.keys(), key=lambda k: abs(k - target))
        if abs(closest_key - target) <= threshold:
            return mapping[closest_key]
        return None






class Act:

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
    act=Act()
    img = act.screenshot()
    state=State()
    state.state_init(img,type="block")
    print(state.grid)

    state.state_refresh()
    print(state.grid)
    '''
    state=State(positions)
    new_matrix=state.state_init()
    print(new_matrix)'''


