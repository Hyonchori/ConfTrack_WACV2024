import numpy as np


class BaseDetection:
    def __init__(
            self,
            xyxy: np.ndarray,  # [x1, y1, x2, y2]
            conf: float,
            cls: int,
            feature: np.ndarray = None,
            det_id: int = None
    ):
        self.xyxy = np.asarray(xyxy, np.float32)
        self.conf = conf
        self.cls = cls
        self.feature = feature / np.linalg.norm(feature) if feature is not None else feature
        self.det_id = det_id

        self.is_matched = False
        self.z, self.width, self.height = self.xyxy2measure()

    def xyxy2measure(self):
        width = max(1., self.xyxy[2] - self.xyxy[0])
        height = max(1., self.xyxy[3] - self.xyxy[1])
        cpwh = [
            [self.xyxy[0] + 0.5 * width],
            [self.xyxy[1] + 0.5 * height],
            [width],
            [height]
        ]
        return np.array(cpwh), width, height  # [center_x, center_y, width, height]

    def get_xyxy(self, padding_ratio=1.0):
        padded_xyxy = self.xyxy.copy()
        if padding_ratio != 1.0:
            padded_xyxy[0] = self.z[0] - self.width * padding_ratio / 2
            padded_xyxy[1] = self.z[1] - self.height * padding_ratio / 2
            padded_xyxy[2] = self.z[0] + self.width * padding_ratio / 2
            padded_xyxy[3] = self.z[1] + self.height * padding_ratio / 2
        return padded_xyxy
