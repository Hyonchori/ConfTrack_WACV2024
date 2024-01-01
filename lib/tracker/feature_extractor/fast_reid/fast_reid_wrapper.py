# BringUp Fast_ReID for extracting feature (not for train)

import os
from pathlib import Path

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import time

from .fastreid.config.config import get_cfg
from .fastreid.modeling.meta_arch.build import build_model
from .fastreid.utils.checkpoint import Checkpointer

FILE = Path(__file__).absolute()


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False

    cfg.freeze()
    return cfg


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


class WrappedFastReID(nn.Module):
    def __init__(
            self,
            cfg_file: str,
            weights_file: str = None,
            device: torch.device = None,
            half: bool = False
    ):
        super().__init__()
        print('\nLoading feature_extractor "FastReID"...')
        print(f'\tcfg_file: {Path(cfg_file).name}, weights_file: {Path(weights_file).name}')
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model, self.cfg = self._init_model(cfg_file, weights_file)
        self.model.eval()

        self.input_size = self.cfg.INPUT.SIZE_TEST
        self.preproc = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.input_size),
            torchvision.transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
            torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        ])

        if half:
            self.model.half()
            print('\tHalf tensor type!')

    @staticmethod
    def _init_model(cfg_file: str, weights_file: str):
        cfg_path = os.path.join(FILE.parents[0], 'configs', 'MOT17', cfg_file)

        if weights_file is not None:
            if not os.path.isfile(weights_file):
                weights_dir = os.path.join(FILE.parents[4], 'pretrained', 'feature_extractor', 'fast_reid')
                weights_file = os.path.join(weights_dir, weights_file)
        else:
            print('\tpretrained extractor weights is None.')

        cfg = setup_cfg(cfg_path, ['MODEL.WEIGHTS', weights_file])
        model = build_model(cfg)
        Checkpointer(model).load(weights_file)
        print(f'\tpretrained extractor weights "{os.path.basename(weights_file)}" are loaded!')
        return model, cfg

    def preprocessing(self, xyxys, img):  # img: torch.Tensor (batch_size, channels, height, width)
        crops = []
        for xyxy in xyxys:
            tmp_crop = self.preproc(
                img[:, :, max(0, int(xyxy[1])): int(xyxy[3]), max(0, int(xyxy[0])): int(xyxy[2])]
            ) * 255
            # tmp_np = tmp_crop.numpy()[0].transpose(1, 2, 0)
            # cv2.imshow('img', tmp_np[..., ::-1].astype(np.uint8))
            # cv2.waitKey(0)
            crops.append(tmp_crop)
        crops = torch.cat(crops).to(self.device).type_as(next(self.model.parameters()))
        return crops

    def forward(self, x, img=None):
        if img is not None:
            x = self.preprocessing(x, img)
        x = self.model(x)
        x = self.postprocessing(x)
        return x

    @staticmethod
    def postprocessing(feats):
        feats[torch.isinf(feats)] = 1.0
        feats = F.normalize(feats)
        return feats.cpu().data.numpy()


def get_wrapped_fast_reid(extractor_cfg, device: torch.device = None):
    if extractor_cfg.extractor_weights == 'fast_reid_mot17':
        cfg_file = 'sbs_S50.yml'
        weights_file = 'mot17_sbs_S50.pth'

    elif extractor_cfg.extractor_weights == 'fast_reid_mot20':
        cfg_file = 'sbs_S50.yml'
        weights_file = 'mot20_sbs_S50.pth'

    elif extractor_cfg.extractor_weights == 'fast_reid_dance':
        cfg_file = 'sbs_S50.yml'
        weights_file = 'dance_sbs_S50.pth'

    else:
        raise Exception(f'Given extractor weights "{extractor_cfg.extractor_weights}" '
                        f'is not valid for wrapped_fast_reid!')

    return WrappedFastReID(
        cfg_file=cfg_file,
        weights_file=weights_file,
        device=device,
        half=extractor_cfg.half
    )
