# Load HiEve dataset in torch.dataloader format

import os
from pathlib import Path
from typing import List
from collections import defaultdict

import natsort
import cv2
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from ..dataset_utils import InferenceTransform


HIEVE_SPLIT = ['train', 'test']
HIEVE_VID = {
    'train': list(range(1, 20)),
    'test': list(range(20, 33))
}


class HiEveDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            use_private_det: bool = False,
            private_det_path: str = None,
            preproc=None,
            use_detector: bool = True,
            use_extractor: bool = True
    ):
        super().__init__()
        self.video_path = video_path
        self.dataset_name = Path(video_path).name
        if not use_private_det and private_det_path is None:
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

        self.cap = cv2.VideoCapture(video_path)
        self.total_len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.det_path is None:
            self.dets = None
        else:
            self.dets = parsing_hieve_detection(self.det_path)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        res, img_raw = self.cap.retrieve()
        if index + 1 == self.total_len:
            self.cap.release()
        if not res:
            raise StopIteration

        if self.dets is not None:
            det = np.array(self.dets[index])
            if len(det) == 0:
                det = np.empty((0, 6))
        else:
            det = np.empty((0, 6))

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


def get_hieve_videos(
        hieve_root: str,  # path to HiEve dataset
        target_split: str,  # select in HIEVE_SPLIT
        target_vid: List[int] = None,  # select in HIEVE_VID
        cfg=None,
        input_size: List[int] = (640, 640)
):
    if target_split == 'train':
        vid_root = os.path.join(hieve_root, 'HIE20', 'videos')
    elif target_split == 'test':
        vid_root = os.path.join(hieve_root, 'HIE20test', 'videos')
    else:
        raise ValueError(f'Given target_split argument is wrong!: {target_split}')
    if not os.path.isdir(vid_root):
        raise ValueError(f'Given video root is wrong!: {vid_root}')

    vid_list = natsort.natsorted(os.listdir(vid_root))
    if target_vid is not None:
        vid_list = [x for x in vid_list if int(x.split('.')[0]) in target_vid]

    print(f'\nLoading videos from HiEve-{target_split}... ')
    preproc = InferenceTransform(img_size=input_size)
    dataloader_kwargs = {
        "num_workers": 1,
        "pin_memory": True,
        "batch_size": 1
    }

    if cfg is not None:
        datasets = [
            HiEveDataset(
                video_path=os.path.join(vid_root, x),
                use_private_det=cfg.use_private_det if hasattr(cfg, 'use_private_det') else False,
                private_det_path=os.path.join(
                    cfg.cfg_path.parents[1],
                    'results',
                    'detector',
                    cfg.type_detector,
                    f'HiEve_{target_split}',
                    cfg.detector_result_dir,
                    f'{x.split(".")[0]}.txt'
                ),
                preproc=preproc,
                use_detector=cfg.use_private_det and not cfg.use_saved_det_result,
                use_extractor=cfg.use_extractor
            ) for x in vid_list
        ]
        # hieve_videos = [
        #     DataLoader(x, collate_fn=x.collate_fn, **dataloader_kwargs) for x in datasets
        # ]
        hieve_videos = datasets
    else:
        hieve_videos = [HiEveDataset(video_path=os.path.join(vid_root, x)) for x in vid_list]

    print(f'\ttotal {len(vid_list)} videos are ready!')
    return hieve_videos


def parsing_hieve_detection(det_path: str):
    with open(det_path) as f:
        rets = [list(map(float, x.strip('\n').split(','))) for x in f.readlines()]
        dets = defaultdict(list)
        for ret in rets:
            dets[int(ret[0]) - 1].append(np.array(ret[2:7] + [0]))
    return dets

