import numpy as np

def iom_nms(boxes, scores, iom_threshold):
    """
    Perform non-maximum suppression using IOM (Intersection over Minimum)
    
    Args:
    - boxes: numpy array of shape (N, 4) representing bounding boxes
             in format [x1, y1, x2, y2]
    - scores: numpy array of shape (N,) representing confidence scores
    - iom_threshold: float, the IOM threshold for suppressing bounding boxes
    
    Returns:
    - numpy array of indices of the boxes to keep
    """
    
    # Sort boxes by score
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # Calculate IOM between the current box and all other boxes
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # Calculate the minimum area
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        min_areas = np.minimum(areas[i], areas[order[1:]])

        # Calculate IOM
        iom = inter / min_areas

        # Keep the boxes with IOM less than the threshold
        inds = np.where(iom <= iom_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)

# Example usage
if __name__ == "__main__":
    # Example bounding boxes: [x1, y1, x2, y2]
    boxes = np.array([
        [100, 100, 200, 200],
        [110, 110, 210, 210],
        [120, 120, 220, 220],
        [130, 130, 230, 230],
        [200, 200, 300, 300]
    ])
    
    # Example scores
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    # IOM threshold
    iom_threshold = 0.5
    
    # Perform NMS
    keep_indices = iom_nms(boxes, scores, iom_threshold)
    
    print("Indices of kept boxes:", keep_indices)
    print("Kept boxes:")
    print(boxes[keep_indices])
    print("Kept scores:")
    print(scores[keep_indices])