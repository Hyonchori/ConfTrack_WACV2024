# Load DanceTrack dataset in torch.dataloader format

import os
from pathlib import Path
from typing import List
from collections import defaultdict

import cv2
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from ..dataset_utils import InferenceTransform

DANCE_SPLIT = ['train', 'val', 'test']
DANCE_VID = {
    'train': [1, 2, 6, 8, 12, 15, 16, 20, 23, 24, 27, 29, 32, 33, 37, 39, 44, 45, 49,
              51, 52, 53, 55, 57, 61, 62, 66, 68, 69, 72, 74, 75, 80, 82, 83, 86, 87, 96, 98, 99],  # total 40
    'val': [4, 5, 7, 10, 14, 18, 19, 25, 26, 30, 34, 35, 41, 43, 47, 58, 63, 65, 73,
            77, 79, 81, 90, 94, 97],  # total 25,
    'test': [3, 9, 11, 13, 17, 21, 22, 28, 31, 36, 38, 40, 42, 46, 48, 50, 54, 56, 59,
             60, 64, 67, 70, 71, 76, 78, 84, 85, 88, 89, 91, 92, 93, 95, 100]  # total 35
}


class DanceTrackDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str,
            use_private_det: bool = False,
            private_det_path: str = None,
            preproc=None,
            use_detector: bool = True,
            use_extractor: bool = True
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_name = Path(dataset_dir).name
        self.img_dir = os.path.join(dataset_dir, 'img1')
        if not use_private_det or private_det_path is None:
            self.det_path = None
        elif use_private_det and not os.path.isfile(private_det_path):
            print(f'\t{private_det_path} is wrong!')
            self.det_path = None
        else:
            print(f'\tprivate detection results are loaded for {self.dataset_name}!')
            self.det_path = private_det_path
        self.preproc = preproc
        self.use_detector = use_detector
        self.use_extractor = use_extractor

        self.img_files = sorted(os.listdir(self.img_dir))
        if self.det_path is None:
            self.dets = {i: [] for i in range(len(self.img_files))}
        else:
            self.dets = parsing_dance_detection(self.det_path)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_files[index])
        det = np.array(self.dets[index])

        img_raw = cv2.imread(img_path)

        if self.use_detector and self.use_extractor:
            if self.preproc is not None:
                img = self.preproc(img_raw)  # for detector
                img_ori_size = self.preproc(img_raw, origin_size=True)  # for extractor
            else:
                img = img_raw.copy()
                img_ori_size = img_raw.copy()
            return img_raw, img, img_ori_size, det

        elif self.use_detector and not self.use_extractor:
            if self.preproc is not None:
                img = self.preproc(img_raw)
            else:
                img = img_raw.copy()
            return img_raw, img, det

        elif not self.use_detector and self.use_extractor:
            if self.preproc is not None:
                img_ori_size = self.preproc(img_raw, origin_size=True)
            else:
                img_ori_size = img_raw.copy()
            return img_raw, img_ori_size, det

        else:  # not self.use_detector and not self.use_extractor
            return img_raw, det

    def collate_fn(self, batch):
        if self.use_detector and self.use_extractor:
            img_raw, img, img_ori_size, det = zip(*batch)
            img = torch.stack([torch.from_numpy(x) for x in img])
            img_ori_size = torch.stack([torch.from_numpy(x) for x in img_ori_size])
            return img_raw, img, img_ori_size, det

        elif self.use_detector and not self.use_extractor:
            img_raw, img, det = zip(*batch)
            img = torch.stack([torch.from_numpy(x) for x in img])
            return img_raw, img, det

        elif not self.use_detector and self.use_extractor:
            img_raw, img_ori_size, det = zip(*batch)
            img_ori_size = torch.stack([torch.from_numpy(x) for x in img_ori_size])
            return img_raw, img_ori_size, det

        else:  # not self.use_detector and not self.use_extractor
            img_raw, det = zip(*batch)
            return img_raw, det


def get_dance_videos(
        dance_root: str,  # path to DanceTrack dataset
        target_split: str,  # select in DANCE_SPLIT
        target_vid: List[int] = None,  # select in DANCE_VID
        cfg=None,
        input_size: List[int] = (640, 640)
):
    vid_root = os.path.join(dance_root, target_split)
    if not os.path.isdir(vid_root):
        raise ValueError(f'Given arguments are wrong!: {dance_root}, {target_split}')

    vid_list = sorted(os.listdir(vid_root))
    if target_vid is not None:
        vid_list = [x for x in vid_list if int(x[-4:]) in target_vid]

    print(f'\nLoading videos from DanceTrack-{target_split}... ')
    if cfg is not None:
        preproc = InferenceTransform(img_size=input_size)
        dataloader_kwargs = {
            "num_workers": 6,
            "pin_memory": True,
            "batch_size": 1
        }
        datasets = [
            DanceTrackDataset(
                dataset_dir=os.path.join(vid_root, x),
                use_private_det=cfg.use_private_det and cfg.use_saved_det_result
                if hasattr(cfg, 'use_private_det') and hasattr(cfg, 'use_saved_det_result') else False,
                private_det_path=os.path.join(
                    cfg.cfg_path.parents[1],
                    'results',
                    'detector',
                    cfg.type_detector,
                    f'DanceTrack_{target_split}',
                    cfg.detector_result_dir,
                    f'{"-".join(x.split("-")[:2])}.txt'
                ),
                preproc=preproc,
                use_detector=cfg.use_private_det and not cfg.use_saved_det_result,
                use_extractor=cfg.use_extractor
            ) for x in vid_list
        ]
        dance_videos = [
            DataLoader(x, collate_fn=x.collate_fn, **dataloader_kwargs) for x in datasets
        ]
    else:
        dance_videos = [DanceTrackDataset(dataset_dir=os.path.join(vid_root, x)) for x in vid_list]

    print(f'\ttotal {len(vid_list)} videos are ready!')
    return dance_videos


def parsing_dance_detection(det_path: str):
    with open(det_path) as f:
        rets = [list(map(float, x.strip('\n').split(','))) for x in f.readlines()]
        dets = defaultdict(list)
        for ret in rets:
            dets[int(ret[0]) - 1].append(ret[2:7] + [0])
    return dets


def parsing_dance_gt(gt_path):
    with open(gt_path) as f:
        rets = [x.strip('\n').split(',') for x in f.readlines()]
        gt = defaultdict(list)
        for ret in rets:
            gt[int(ret[0])].append(ret[1:])
    return gt
