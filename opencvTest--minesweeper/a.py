import cv2
import numpy as np
import os
import mss
import time

thresh_IoU = 0.32
threshold = 0.57  # 可临时调低测试

def preprogress(img):
    """统一预处理：BGR -> 灰度 -> 高斯模糊 -> Otsu 二值化"""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_binary = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_binary

def match(template, processed_img, original_bgr):
    w = template.shape[1]
    h = template.shape[0]
    result = cv2.matchTemplate(processed_img, template, cv2.TM_CCOEFF_NORMED)

    # 调试：查看分数范围
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print(f"匹配分数范围: {min_val:.3f} ~ {max_val:.3f}")

    loc = np.where(result > threshold)
    print(f"高于阈值 {threshold} 的像素点数量: {len(loc[0])}")

    if len(loc[0]) == 0:
        print("未找到匹配")
        return original_bgr

    boxes = np.column_stack([loc[1], loc[0], loc[1]+w, loc[0]+h])
    scores = result[loc]

    # NMS
    max_boxes = []
    while boxes.size > 0:
        index = np.argmax(scores)
        max_boxes.append(boxes[index])
        scores = np.delete(scores, index)
        boxes = np.delete(boxes, index, axis=0)

        to_delete = []
        for i, box in enumerate(boxes):
            iou = IoU(max_boxes[-1], box)
            if iou > thresh_IoU:
                to_delete.append(i)

        mask = np.ones(len(boxes), dtype=bool)
        mask[to_delete] = False
        boxes = boxes[mask]
        scores = scores[mask]

    # 绘制结果
    img_copy = original_bgr.copy()
    for box in max_boxes:
        cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    return img_copy

def IoU(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    intersection = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box_area1 = (x2 - x1) * (y2 - y1)
    box_area2 = (x4 - x3) * (y4 - y3)
    union = box_area1 + box_area2 - intersection
    if union == 0:
        return 0
    return intersection / union

# 主程序
base_dir = os.path.dirname(os.path.abspath(__file__))
template_num1 = cv2.imread(os.path.join(base_dir, "temp", "num1.png"))

with mss.mss() as sct:
    monitor = {"top": 284, "left": 280, "width": 1910, "height": 1313}
    time.sleep(3)
    img = np.array(sct.grab(monitor))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 转为 BGR

    processed_img = preprogress(img_bgr)
    processed_template_num1 = preprogress(template_num1)

    result_img = match(processed_template_num1, processed_img, img_bgr)
    cv2.imshow("检测结果", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()