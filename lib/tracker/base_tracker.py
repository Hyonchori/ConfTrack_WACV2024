from typing import List

import torch
import numpy as np

from .detection.base_detection import BaseDetection
from .kalman_filter.config_kalmanfilter import get_kalmanfilter
from .matching.config_matching import get_matching_fn
from .track.config_track import BaseTrack, get_track
from .feature_extractor.config_extractor import get_extractor
from .gmc.base_cmc import BaseCMC


class BaseTracker:
    def __init__(self, trk_cfg, device=None):
        print(f'\nLoading tracker "{trk_cfg.tracker_name}"...')
        self.cfg = trk_cfg
        self.matching_fn = get_matching_fn(self.cfg)
        self.extractor = get_extractor(self.cfg, device) if self.cfg.use_extractor else None

        self.tracks: List[BaseTrack] = []
        self.track_id = 1
        self.cmc = None

    def initialize(self, vid_name: str = None, target_split: str = None):
        self.tracks = []
        self.track_id = 1

        if self.cfg.use_cmc and vid_name is not None:
            self.cmc = BaseCMC(
                type_cmc=self.cfg.type_cmc,
                cmc_result_dir=self.cfg.cmc_result_dir,
                vid_name=vid_name,
                target_split=target_split
            )

    def init_tracks(self, detections: List[BaseDetection], unmatched_det_indices: List[int]):
        for det_idx in unmatched_det_indices:
            det = detections[det_idx]
            if det.conf < self.cfg.track_new_thr:
                continue
            kf = get_kalmanfilter(self.cfg, det.z, det.conf, det.cls)
            track = get_track(self.cfg, self.track_id, det, kf)
            self.tracks.append(track)
            self.track_id += 1

    def predict(self, raw_frame: np.ndarray = None, detections: List[BaseDetection] = None, img_idx: int = None):
        if self.cfg.use_cmc and self.cmc is not None:
            affine_matrix = self.cmc.compute_affine(raw_frame, detections, img_idx)

        for track in self.tracks:
            track.predict()
            if self.cfg.use_cmc and self.cmc is not None:
                track.apply_affine(affine_matrix)

    def update(self, detections: List[BaseDetection], img_for_extractor: torch.Tensor = None):
        matches, unmatched_trk_indices, unmatched_det_indices = self.matching_fn(
            cfg=self.cfg,
            trk_list=self.tracks,
            det_list=detections,
            img_for_extractor=img_for_extractor,
            extractor=self.extractor
        )

        for trk_idx, det_idx in matches:
            tmp_trk = self.tracks[trk_idx]
            tmp_det = detections[det_idx]
            tmp_trk.measure(tmp_det)

        for track in self.tracks:
            track.update()

        self.init_tracks(detections, unmatched_det_indices)
        deleted_trk_indices = [track.track_id for track in self.tracks if track.is_ambiguous()]
        self.tracks = [track for track in self.tracks if not track.is_ambiguous()]

        return matches, unmatched_trk_indices, unmatched_det_indices, deleted_trk_indices
