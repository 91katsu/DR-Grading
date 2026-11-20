'''
2025/07/25
Version 2.1.6: #1901 update ALFA label_max (解決後處理拉起confident score過低的框)
----------
請填寫日期、後處理版本
'''
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import copy
from .nms import iom_nms
from .adjust_box import adjust_bbox
from .ALFA import contradict_DR5
'''
Return the prediction result after post=processing

Parameters
----------
boxes: list
    A list of bounding boxes, where each bounding box is represented as a list of four values 
    [x_min, y_min, x_max, y_max], defining the coordinates of the detected object in the image.
scores: list
    The confidence scores for all bounding boxes. Each score is a float representing the model's confidence 
    that the detected object belongs to the predicted class.
labels: list
    A list of labels corresponding to the predicted class for each bounding box. 
    Each label represents the class index for the detected object.
    
score_threshold
    The threshold value for score used to determine whether a bounding box should be retained.

crop_info: dict
    The cropping information during pre-processing (How much was cropped from the top, bottom, left and right)
padding_info: dict
    The padding information during pre-processing (How much was padded from the top, bottom, left and right)

DR_class: int
    Integer representing the predicted class index. (0 represents DR0, 1 represents DR1, and so on).
probability: float
    The probability of predicted classes.

image_shape: list
    The shape of original fundus image.

returns
----------
boxes: list
    A list of bounding boxes based on relative position after post-processing
scores: list
    The confidence scores for boxes retained after post-processing
labels: list
    A list of labels for each boxes retained after post-processing
cls: int
    The classification result after post-processing

Examples
--------
>>> adjusted_boxes, nms_scores, nms_labels, cls = postprocess(boxes, scores, labels, 0.3, crop_info, padding_info, classifi_prediction, prediction_probability, img_shape)
>>> print(adjusted_boxes, nms_scores, nms_labels, cls, sep='\n')
[[0.6205343606834588, 0.9068402783624057, 0.09324726860644081, 0.06625108390018858], [0.5821997646911903, 0.10506623531210012, 0.0856574977048531, 0.08802047926804114], [0.6084181438393307, 0.22576489941827182, 0.040731087258334533, 0.06387373825599407], [0.5298814202233944, 0.3259231041217672, 0.017122541155133928, 0.025262661637931033], [0.6414347881545669, 0.3416028779128502, 0.053296453941802276, 0.12106107514480065], [0.49608211253645235, 0.9165946434284079, 0.04529940697454637, 0.0666448658910291], [0.5833002257456977, 0.7920850556472252, 0.016540105441748273, 0.024637366985452585], [0.6818766615907168, 0.46719478738719017, 0.0810619653095298, 0.09307024725552263], [0.6779548891129032, 0.7760288106984106, 0.03216995731476815, 0.04325361580684267], [0.7176386749689481, 0.27237333100417566, 0.09224373953683036, 0.11733182709792565], [0.3653911423573296, 0.15803705412766028, 0.1417717032718219, 0.14636446196457437], [0.7546317104919715, 0.3677638349861934, 0.03667080786920363, 0.051105420342807115], [0.7475618283320132, 0.6548732626027074, 0.03981088383406538, 0.05901947021484375], [0.7216539163193945, 0.7227830821070178, 0.03289654287874424, 0.06384177372373384], [0.5808344194965978, 0.43041665965113146, 0.01570734120733727, 0.02499579067887931], [0.722571289484402, 0.794891567887931, 0.02733794884747624, 0.041053298424030174], [0.7342031804097963, 0.5413281144766972, 0.028865831788234448, 0.03867776804956897], [0.6933946038171444, 0.5960846736513336, 0.030105379869311635, 0.03692358608903556], [0.4971999067315308, 0.15001988575376313, 0.028309835266957087, 0.040851204970787314], [0.6965289313672325, 0.6463557802397629, 0.021415499498217885, 0.031716392780172416]]
[0.6104668378829956, 0.6009312272071838, 0.5963906645774841, 0.5769655108451843, 0.5741189122200012, 0.5457523465156555, 0.5275290012359619, 0.5223801732063293, 0.5189064145088196, 0.5142690539360046, 0.506237804889679, 0.39234134554862976, 0.37936142086982727, 0.36079514026641846, 0.3458285927772522, 0.3430447578430176, 0.3153768479824066, 0.3044009804725647, 0.3012661933898926, 0.3008103370666504]
[12, 12, 12, 1, 12, 12, 1, 12, 12, 12, 12, 12, 12, 12, 1, 12, 12, 12, 8, 8]
4
'''
def postprocess(boxes, scores, labels, score_threshold, crop_info, padding_info, DR_class, probability, image_shape, type=1):



    if type==0:
        # JYE01
        cls_prob_path = "./postprocess/cls_prob_jye01.csv"
        score_path = "./postprocess/Score_jye01.csv"        
    else:
        # Nidek, Canon, ...
        cls_prob_path = "./postprocess/cls_prob.csv"
        score_path = "./postprocess/Score.csv"

    keep_indices = iom_nms(np.array(boxes), np.array(scores), iom_threshold=0.5)
    if len(keep_indices) != 0:
        nms_boxes = np.array(boxes)[keep_indices].tolist()
        nms_labels = np.array(labels)[keep_indices].tolist()
        nms_scores = np.array(scores)[keep_indices].tolist()

        adjusted_boxes = [
            adjust_bbox(bbox, crop_info, padding_info) for bbox in nms_boxes
        ]
        relative_boxes = []

        for box in adjusted_boxes:
            x1, y1, x2, y2 = box
            w, h = image_shape[1], image_shape[0]
            relative_box = [(x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h]
            relative_boxes.append(relative_box)
        # print(relative_boxes, nms_scores, nms_labels)
        
    else:
        relative_boxes = []
        nms_scores = []
        nms_labels = []


    all_detections = {"boxes": np.array(relative_boxes),
                      "scores": np.array(nms_scores),
                      "labels": np.array(nms_labels)} 

    cls, det = contradict_DR5(copy.deepcopy(DR_class), copy.deepcopy(probability), copy.deepcopy(all_detections), score_threshold, image_shape, cls_prob_path, score_path)

    high_score_indices = np.zeros(len(det["labels"]), dtype=bool)
    for i in range(1, 14):  # Assuming labels range from 1 to 13
        tmp = (np.array(det["labels"]) == i) & (np.array(det["scores"]) > score_threshold.get(i, 0.3))
        high_score_indices |= tmp

    high_score_indices = np.where(high_score_indices)[0]

    # # list to numpy array
    socre_boxes = np.array(det["boxes"])[high_score_indices]
    socre_scores = np.array(det["scores"])[high_score_indices]
    socre_labels = np.array(det["labels"])[high_score_indices]

    return socre_boxes, socre_scores, socre_labels, cls


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    box1, box2 : list or array-like
        Bounding boxes in [x1, y1, x2, y2] format.

    Returns
    -------
    float
        IoU value in [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def eliminate_spot_light(spot_light_results, lesion_boxes, lesion_scores, lesion_labels, crop_info, padding_info,
                         spot_light_threshold=0.5, iou_threshold=0.3):
    """
    Remove lesion detections that are likely caused by fixed light spots.

    For each spot light detection whose score exceeds `spot_light_threshold`,
    any lesion box whose IoU with that spot light box exceeds `iou_threshold`
    is removed.

    Parameters
    ----------
    spot_light_results : dict
        Output from the spot light detector.
        Expected keys:
            'boxes'  – list of [x1, y1, x2, y2] (normalised, xyxyn format)
            'scores' – list of float confidence scores
    lesion_boxes : numpy.ndarray or list
        Lesion bounding boxes in [cx, cy, w, h] normalised format.
    lesion_scores : numpy.ndarray or list
    lesion_labels : numpy.ndarray or list

    spot_light_threshold : float
        Minimum spot light score to trigger suppression.
    iou_threshold : float
        IoU threshold above which a lesion box is suppressed.

    Returns
    -------
    lesion_boxes, lesion_scores, lesion_labels : numpy.ndarray
        Filtered detection results.
    """
    if len(lesion_boxes) == 0:
        return lesion_boxes, lesion_scores, lesion_labels

    # print('lesion_boxes', lesion_boxes)
    # print('lesion_scores', lesion_scores)
    # print('lesion_labels', lesion_labels)
    lesion_boxes   = np.array(lesion_boxes)
    lesion_scores  = np.array(lesion_scores)
    lesion_labels  = np.array(lesion_labels)

    sl_boxes  = spot_light_results.get('boxes', [])
    sl_scores = spot_light_results.get('scores', [])

    keep = np.ones(len(lesion_boxes), dtype=bool)

    for sl_box, sl_score in zip(sl_boxes, sl_scores):
        if sl_score <= spot_light_threshold:
            continue
        for i, lesion_box in enumerate(lesion_boxes):
            if not keep[i]:
                continue
            iou = calculate_iou(sl_box, lesion_box)
            if i==0:
                print(sl_box, lesion_box, iou)
            if iou > iou_threshold:
                keep[i] = False
    return np.array(lesion_boxes)[keep], lesion_scores[keep], lesion_labels[keep]
