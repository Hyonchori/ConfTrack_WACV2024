from typing import List
import numpy as np

from .base_detection import BaseDetection


def get_detection(
        xyxy: np.ndarray,  # [x1, y1, x2, y2]
        conf: float,
        cls: int,
        feature: np.ndarray = None,
        det_id: int = None
):
    return BaseDetection(
        xyxy=xyxy,
        conf=conf,
        cls=cls,
        feature=feature,
        det_id=det_id
    )


def get_detections(
        cfg,
        predictions: np.ndarray,  # detector output: [x1, y1, x2, y2, confidence, class]
        target_classes: int = 0
) -> List[BaseDetection]:
    detections = []

    for res in predictions:
        if len(res) == 7:
            conf = float(res[4]) * float(res[5])
        else:  # len(res) == 6:
            conf = float(res[4])
        if conf < cfg.detection_low_thr:
            continue
        bbox = res[:4]
        cls = int(res[-1])
        if target_classes is not None and cls != target_classes:
            continue
        detections.append(
            get_detection(xyxy=bbox, conf=conf, cls=cls)
        )

    return detections
