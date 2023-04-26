def calculate_iou(a: list, b: list) -> float:
    """
    calculate intersection of two temporal segments
    """
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = min(max(a[1], b[1]) - min(a[0], b[0]), a[1] - a[0] + b[1] - b[0])
    iou = float(intersection) / (union + 1e-8)

    return iou
