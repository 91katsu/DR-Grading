def adjust_bbox(bbox, crop_info, padding_info):
    x1, y1, x2, y2 = bbox

    # 移除填充
    x1 -= padding_info["pad_left"]
    y1 -= padding_info["pad_top"]
    x2 -= padding_info["pad_left"]
    y2 -= padding_info["pad_top"]

    # 還原裁剪
    x1 += crop_info["left"]
    y1 += crop_info["top"]
    x2 += crop_info["left"]
    y2 += crop_info["top"]

    return [float(x1), float(y1), float(x2), float(y2)]
