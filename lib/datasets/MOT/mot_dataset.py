# Load MOT dataset in torch.dataloader format

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


MOT_SELECTION = ['MOT17', 'MOT20']
MOT_SPLIT = ['train', 'val', 'test']
MOT_VID = {
    'MOT17':
        {'train': [2, 4, 5, 9, 10, 11, 13],  # total 7
         'val': [2, 4, 5, 9, 10, 11, 13],  # total 7
         'test': [1, 3, 6, 7, 8, 12, 14]},  # total 7
    'MOT20':
        {'train': [1, 2, 3, 5],  # total 4
         'val': [1, 2, 3, 5],  # total 4
         'test': [4, 6, 7, 8]}  # total 4
}
MOT17_DET = ['DPM', 'FRCNN', 'SDP']
MOT_CLASSES = {0: 'pedestrian', 1: 'person_on_vehicle', 2: 'car', 3: 'bicycle', 4: 'motorbike',
               5: 'non_motorized_vehicle', 6: 'static_person', 7: 'distractor', 8: 'occluder',
               9: 'occluder_on_the_ground', 10: 'occluder_full', 11: 'reflection', 12: 'crowd'}


class MOTDataset(Dataset):
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
        if not use_private_det and private_det_path is None:
            self.det_path = os.path.join(dataset_dir, 'det', 'det.txt')
        elif use_private_det and not os.path.isfile(private_det_path):
            print(f'\t{private_det_path} is wrong!')
            self.det_path = os.path.join(dataset_dir, 'det', 'det.txt')
        else:
            print(f'\tprivate detection results are loaded for {self.dataset_name}!')
            self.det_path = private_det_path
        self.preproc = preproc
        self.use_detector = use_detector
        self.use_extractor = use_extractor

        self.img_files = sorted(os.listdir(self.img_dir))
        self.dets = parsing_mot_detection(self.det_path)

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


def get_mot_videos(
        mot_root: str,  # path to MOT dataset
        target_select: str,  # select in MOT_SELECTION
        target_split: str,  # select in MOT_SPLIT
        target_vid: List[int] = None,  # select in MOT_VID
        target_det: List[str] = None,  # for MOT17, select in MOT17_DET
        cfg=None,
        input_size: List[int] = (640, 640)
):
    vid_root = os.path.join(mot_root, target_select, target_split)
    if not os.path.isdir(vid_root):
        raise ValueError(f'Given arguments are wrong!: {mot_root}, {target_select}, {target_split}')

    vid_list = sorted(os.listdir(vid_root))
    if target_vid is not None:
        vid_list = [x for x in vid_list if int(x.split('-')[1]) in target_vid]

    if target_select == 'MOT17' and target_det is not None:
        vid_list = [x for x in vid_list if x.split('-')[-1] in target_det]
        remain_dets = [x for x in MOT17_DET if not x in target_det]
    elif target_select == 'MOT17' and target_det is None:
        vid_list = [x for x in vid_list if x.split('-')[-1] == 'FRCNN']
        remain_dets = ['DPM', 'SDP']
    else:
        remain_dets = []

    print(f'\nLoading videos from {target_select}-{target_split}... ')
    preproc = InferenceTransform(img_size=input_size)
    dataloader_kwargs = {
        "num_workers": 6,
        "pin_memory": True,
        "batch_size": 1
    }
    if cfg is not None:
        datasets = [
            MOTDataset(
                dataset_dir=os.path.join(vid_root, x),
                use_private_det=cfg.use_private_det and cfg.use_saved_det_result
                if hasattr(cfg, 'use_private_det') and hasattr(cfg, 'use_saved_det_result') else False,
                private_det_path=os.path.join(
                    cfg.cfg_path.parents[1],
                    'results',
                    'detector',
                    cfg.type_detector,
                    f'{target_select}_{target_split}',
                    cfg.detector_result_dir,
                    f'{"-".join(x.split("-")[:2])}.txt',
                ) if cfg.detector_result_dir is not None and cfg.use_saved_det_result else None,
                preproc=preproc,
                use_detector=cfg.use_private_det and not cfg.use_saved_det_result,
                use_extractor=cfg.use_extractor
            ) for x in vid_list
        ]
        mot_videos = [
            DataLoader(x, collate_fn=x.collate_fn, **dataloader_kwargs) for x in datasets
        ]
    else:
        mot_videos = [MOTDataset(dataset_dir=os.path.join(vid_root, x)) for x in vid_list]

    print(f'\ttotal {len(vid_list)} videos are ready!')
    return mot_videos, remain_dets


def parsing_mot_detection(det_path: str):
    with open(det_path) as f:
        rets = [list(map(float, x.strip('\n').split(','))) for x in f.readlines()]
        dets = defaultdict(list)
        for ret in rets:
            dets[int(ret[0]) - 1].append(np.array(ret[2:7] + [0]))
    return dets


def parsing_mot_gt(gt_path: str):
    with open(gt_path) as f:
        rets = [x.strip('\n').split(',') for x in f.readlines()]
        gt = defaultdict(list)
        for ret in rets:
            gt[int(ret[0]) - 1].append(np.array(ret[1:]))
    return gt
