import os
import shutil
from pathlib import Path

FILE = Path(__file__).absolute()


class TrackerCFG:
    def __init__(self):
        self.tracker_name = 'ConfTrack'
        self.cfg_path = FILE
        self.save_only_matched = True
        self.save_only_confirmed = True

        # contributions of ConfTrack
        self.use_CWKU = True  # use Confidence Weighted Kalman Update
        self.use_NSAK = True  # use Noise Scale Adaptive Kalman Filter
        self.use_CPLT = True  # use Constant Prediction on Lost Track
        self.use_CFCM = True  # use Confidence Fused Cost Matrix
        self.use_LCTM = True  # use Low Confident Track Matching

        # attributes for detector
        self.type_detector = 'yolox'
        self.detector_input_size = None
        self.detector_conf_thr = 0.01
        self.detector_iou_thr = 0.7
        self.detector_weights = 'yolox_x_byte_ablation'
        self.use_private_det = True
        self.use_saved_det_result = True
        self.detector_result_dir = 'yolox_x_byte_mot17_ablation'
        self.fuse = True
        self.half = True

        # attributes for detection
        self.detection_low_thr = 0.1
        self.detection_high_thr = 0.6

        # attributes for Kalman filter
        self.std_weight_position = 1. / 20
        self.std_weight_velocity = 1. / 160
        self.nsa_amplify_factor = 100.

        # attributes for track
        self.track_new_thr = 0.1  # confidence threshold for new track initialization
        self.track_confirm_thr = 0.7  # confidence threshold for track confirmation
        self.max_age = 30
        self.init_age = 3
        self.ema_alpha = 0.9
        self.aspect_ratio_thr = 1.6
        self.area_thr = 100
        self.only_pedestrian = True
        self.confidence_decay = 0.9

        # attributes for track-detection matching
        self.type_matching = 'conftrack'
        self.first_matching_iou_thresh = 0.6
        self.first_matching_emb_thresh = 0.25
        self.first_matching_thresh = 0.8
        self.second_matching_iou_thresh = 0.5
        self.low_confidence_matching_negli_thresh = 0.7
        self.low_confidence_matching_thresh = 0.3

        # attributes for feature extractor
        self.type_extractor = 'fast_reid'
        self.extractor_weights = 'fast_reid_mot17'
        self.use_extractor = True

        # attributes for CMC(camera motion compensation)
        self.type_cmc = 'file'  # ['file', 'sparse', 'sift', 'ecc', None]
        self.cmc_result_dir = os.path.join(self.cfg_path.parents[0], '../cmc_files')
        self.use_cmc = True

    def save_opt(self, save_dir):
        save_path = os.path.join(save_dir, self.cfg_path.name)
        shutil.copyfile(self.cfg_path, save_path)
