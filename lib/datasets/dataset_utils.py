# util functions for dataset loading

import math
from typing import List

import cv2
import numpy as np


def preproc(image, input_size):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    return padded_img


def letterbox(
        img: np.ndarray,
        new_shape=(640, 640),
        color: int = (114., 114., 114.),
        auto: bool = True,
        stretch: bool = False,
        stride: int = 32,
        dnn_pad: bool = False,
        center_focus: bool = False,
):
    # resize and pad image while meeting stride-multiple constraints
    shape = img.shape[: 2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    if dnn_pad:
        new_shape = [stride * math.ceil(x / stride) for x in new_shape]

    if img.shape[:2] == new_shape:
        return img, 1., (0, 0)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif stretch:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    if center_focus:
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    else:
        top, bottom = 0, int(round(dh + 0.1))
        left, right = 0, int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img, ratio, (dw, dh)


class InferenceTransform:
    def __init__(
            self,
            img_size: List[int] = (640, 640),
            rgb_means: List[float] = (0.485, 0.456, 0.406),
            rgb_std: List[float] = (0.229, 0.224, 0.225),
            swap: List[int] = (2, 0, 1),
            bgr2rgb: bool = True,
            scaling: bool = True,
            normalize: bool = True
    ):
        self.img_size = img_size
        self.rgb_means = rgb_means
        self.rgb_std = rgb_std
        self.swap = swap
        self.bgr2rgb = bgr2rgb
        self.scaling = scaling
        self.normalize = normalize

    def __call__(self, img: np.ndarray, origin_size: bool = False):
        # resize and padding
        if origin_size:
            img = img.copy().astype(np.float64)
        else:
            # img = preproc(img, self.img_size)
            img = letterbox(img, self.img_size, auto=False, dnn_pad=False)[0].astype(np.float32)

        # bgr2rgb
        if self.bgr2rgb:
            img = img[:, :, ::-1]

        # scaling [0 ~ 255] -> [0.0 ~ 1.0]
        if self.scaling:
            img /= 255.0

        # normalize
        if self.normalize:
            img -= self.rgb_means
            img /= self.rgb_std

        # swap channels [height, width, channels] -> [channels, height, width]
        img = img.transpose(self.swap)

        # make contiguous array form faster inference
        img = np.ascontiguousarray(img, np.float32)

        return img
