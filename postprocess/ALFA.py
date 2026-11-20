import numpy as np
import pandas as pd
import copy

"""
ALFA: Adaptive Lesion-based Feature Algorithm.
0.0.1: #1589 update ALFA of DR2/DR3 threshold
0.0.2: #1602 update det_theshold for each feature
0.0.3: #1630 update theshold by 1112 Detr model
0.0.4: #1642 update thresholds_dict by class
0.0.5: #1650 update to swin v2 base
0.0.6: #1653 simplify
1.0.0: #1668 separate JYE01; read csv as each threshold value
1.0.1: #1678 update csv according to "swinv2_l_384_tung0116_v5-4_20250116_crop"
1.0.2: #1689 turn off Cls==3
1.0.3: #1704 turn on Cls==3
1.0.4: #1832 update JYE010325
1.0.5: #1888 update
1.0.6: #1901 update label_max
"""
def get_box_quadrant(boxes, image_center):
    Cx, Cy = image_center
    quadrants = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        bx = (x1 + x2) / 2
        by = (y1 + y2) / 2

        # 判斷象限
        if bx > Cx and by < Cy:
            quadrant = 1  
        elif bx < Cx and by < Cy:
            quadrant = 2  
        elif bx < Cx and by > Cy:
            quadrant = 3  
        else:
            quadrant = 4  

        quadrants.append(quadrant)
    
    return quadrants

def det2cls(det_output, thresholds_dict, image_shape):
    lesions = []
    boxes = []
    for label, score, box in zip(det_output["labels"], det_output["scores"], det_output["boxes"]):
        if score >= thresholds_dict.get(label, 0.3):  # 如果沒有對應的閾值，使用默認值0.3
            lesions.append(label)
            boxes.append(box)

    lesions = np.array(lesions)
    boxes = np.array(boxes)
    center_x, center_y = image_shape[1] // 2, image_shape[0] // 2
    quadrants_VB = get_box_quadrant(boxes[lesions == 3], (center_x, center_y))

    IH = True if np.sum(lesions == 2) >= 10 else False

    if any(np.isin(np.unique(lesions), [5, 6, 7])):
        cls = 4
        score = np.max(np.array(det_output["scores"])[np.isin(det_output["labels"], [5, 6, 7])])
    elif any(np.unique(lesions) == 4) or len(np.unique(quadrants_VB)) >= 1 or IH or any(np.unique(lesions) == 12):
        cls = 3
        score = np.max(np.array(det_output["scores"])[np.isin(det_output["labels"], [4,3,2,12])])
    elif any(np.isin(np.unique(lesions), [8, 10, 6, 2])):
        cls = 2
        score = np.max(np.array(det_output["scores"])[np.isin(det_output["labels"], [8, 10, 6, 2])])
    elif any(np.unique(lesions) == 1):
        cls = 1
        score = np.max(np.array(det_output["scores"])[np.isin(det_output["labels"], [1])])
    else:
        cls = 0
        if any(np.isin(det_output["labels"], [1,2,3,4,5,6,7,8,10,12])):
            score = 1-np.max(np.array(det_output["scores"])[np.isin(det_output["labels"], [1,2,3,4,5,6,7,8,10,12])])
        else: score = 1

    return cls, score

rows, cols = 5, 5
index_array = [[{} for _ in range(cols)] for _ in range(rows)]
lesion_array = [[] for _ in range(rows)]
lesion_array[1] = [1]
lesion_array[2] = [8, 10, 6, 2]
lesion_array[3] = [4, 3, 2, 12]
lesion_array[4] = [5, 6, 7]

label_max_array = [0,0.1,0.1,0.2,0.2]
def csv2array(cls_prob_path, score_path):
    cls_prob = pd.read_csv(cls_prob_path, header=None)
    cls_prob_list = cls_prob.values.tolist()
    score = pd.read_csv(score_path, header=None)
    score_list = score.values.tolist()
    index_array = [[{} for _ in range(cols)] for _ in range(rows)]
    # print(data_as_list)
    for i in range(5):
        for j in range(5):
            index_array[i][j]["cls_prob"] = cls_prob_list[i][j]
            index_array[i][j]["Score"] = score_list[i][j]
    return index_array

def contradict_DR5(cls_output, cls_prob, det_output, det_theshold, image_shape,
                   cls_prob_path=r'D:\2024_Fundus\code\contradiction\cls_prob.csv',
                   score_path=r'D:\2024_Fundus\code\contradiction\Score.csv'):
    Cls = cls_output 
    Det, Score = det2cls(det_output, det_theshold, image_shape)
    
    # 【修正 2】深拷貝 det_output，確保原始檢測結果不會被迴圈內的拔除操作污染
    det = copy.deepcopy(det_output) 
    
    index_array = csv2array(cls_prob_path, score_path)
    while Cls!=Det:
        if (Cls==4) : 
            Det = Cls

        elif Cls < Det:
            if (cls_prob <= index_array[Cls][Det]["cls_prob"]) & (Score > index_array[Cls][Det]["Score"]):
                Cls = Det
            else: 
                # 1. 複製 threshold 字典，避免污染其他圖片的推論
                local_thresholds = det_theshold.copy()
                labels = np.array(det['labels'])
                scores = np.array(det['scores'])
                
                current_ih_thresh = local_thresholds.get(2, 0.3)
                ih_count = np.sum((labels == 2) & (scores >= current_ih_thresh))

                # 2. 如果 Det==3 且原因是 IH 數量 >= 10，啟動「動態閥值提升」
                if Det == 3 and ih_count >= 10:
                    while ih_count >= 10 and current_ih_thresh < 1.0:
                        current_ih_thresh *= 1.1
                        ih_count = np.sum((labels == 2) & (scores >= current_ih_thresh))
                    
                    # 更新 local 閥值並重新計算 Det
                    local_thresholds[2] = current_ih_thresh
                    det_theshold = local_thresholds
                    Det, Score = det2cls(det, det_theshold, image_shape)

                # 3. Fallback: 拔除病灶
                if Cls < Det:
                    target_lesions = lesion_array[Det].copy()
                    
                    # 【修正 1】只有在 Det == 3 時，才將 IH (2) 從拔除清單中保護起來。
                    # 如果已經是 Det == 2 且 Cls == 1，就必須允許拔除 IH，避免無窮迴圈。
                    if Det == 3 and 2 in target_lesions:
                        target_lesions.remove(2) 
                    
                    if len(target_lesions) > 0:
                        for key in ['boxes', 'scores', 'labels']:
                            det[key] = (np.array(det[key])[np.isin(det['labels'], target_lesions, invert=True)]).tolist()
                    
                    Det, Score = det2cls(det, det_theshold, image_shape)        
        elif Cls > Det:
            if Cls==3: Cls_3 = 3
            else: Cls_3 = None
            if (cls_prob <= index_array[Cls][Det]["cls_prob"]) & (Score > index_array[Cls][Det]["Score"]):   
                Cls = Det
            else: 
                if any(np.isin(det['labels'], lesion_array[Cls])):
                    scores_cls = np.array(det['scores'])[np.isin(det['labels'], lesion_array[Cls])]
                    labels_cls = np.array(det['labels'])[np.isin(det['labels'], lesion_array[Cls])]
                    label_max_lesion = int(labels_cls[np.argmax(scores_cls)])
                    label_max = np.max(scores_cls)
                    if label_max>=det_theshold.get(label_max_lesion, label_max_array[Cls])*0.9:
                        tmp = np.array(det['scores'])
                        for i in lesion_array[Cls]:
                            tmp[np.isin(det['labels'], [i])] += (det_theshold.get(i, 0.3)-np.floor(label_max*100)/100)
                        det['scores'] = tmp.tolist()
                    else: 
                        Cls = Det
                else:
                    Cls = Det
            Det, Score = det2cls(det, det_theshold, image_shape)   
            if (Cls==Cls_3): Cls=Det

    return Cls, det