import numpy as np

from ..detection.base_detection import BaseDetection
from ..kalman_filter.base_kalmanfilter import BaseKalmanFilter


class TrackState:
    Tentative = 1
    Ambiguous = 2
    Confirmed = 3
    Rematched = 4
    Lost = 5


class BaseTrack:
    def __init__(
            self,
            track_id: int,
            detection: BaseDetection,
            kalman_filter: BaseKalmanFilter,
            max_age: int = 30,
            init_age: int = 3,
            ema_alpha: float = 0.9,
            aspect_ratio_thr: float = 1.6,
            area_thr: int = 100,
            only_pedestrian: bool = True,
            use_CWKU: bool = False,
            CWKU_thr: float = 0.6,
            use_CPLT: bool = False,
            track_confirm_thr: float = 0.7,
            confidence_decay: float = 0.9
    ):
        self.track_id = track_id
        self.kf = kalman_filter
        self.max_age = max_age
        self.init_age = init_age
        self.aspect_ratio_thr = aspect_ratio_thr
        self.area_thr = area_thr
        self.only_pedestrian = only_pedestrian
        self.width = detection.width
        self.height = detection.height
        self.init_conf = detection.conf
        self.conf_history = [detection.conf]
        self.conf = detection.conf
        self.cls = detection.cls
        self.use_CWKU = use_CWKU
        self.CWKU_thr = CWKU_thr
        self.use_CPLT = use_CPLT
        self.track_confirm_thr = track_confirm_thr
        self.confidence_decay = confidence_decay

        if detection.feature is not None:
            self.smooth_feat = detection.feature / np.linalg.norm(detection.feature)
        else:
            self.smooth_feat = None
        self.ema_alpha = ema_alpha

        self.x, self.x_cov = self.kf.initialize_state(self.width, self.height)
        self.is_matched = False
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
        self.track_state = TrackState.Tentative

    def state2wh(self):
        return self.x[2][0], self.x[3][0]

    def get_projected_state(self):
        return self.kf.project(self.width, self.height)

    def predict(self):
        self.is_matched = False
        self.age += 1
        self.time_since_update += 1
        if self.is_confirmed() or self.is_lost():
            self.x, self.x_cov = self.kf.predict(self.width, self.height,
                                                 use_CPLT=self.use_CPLT and self.is_lost())

        self.width, self.height = self.state2wh()

    def apply_affine(self, affine_matrix: np.ndarray):  # shape: (2, 3)
        rotation_mat = affine_matrix[:2, :2]
        translation_mat = affine_matrix[:2, 2:3]
        tmp_cp = self.x[:2].copy()
        comp_cp = np.matmul(rotation_mat, tmp_cp) + translation_mat
        self.x[:2] = comp_cp
        self.width, self.height = self.state2wh()
        self.kf.x = self.x

    def measure(self, detection: BaseDetection):
        self.is_matched = True

        if self.use_CWKU and detection.conf < self.CWKU_thr:
            ''' Confidence Weighted Kalman Update '''
            dz = detection.z - self.x[:4]
            target_measure = self.x[:4] + dz * detection.conf
        else:
            target_measure = detection.z
        self.kf.measure(target_measure, detection.conf)
        self.conf = detection.conf
        self.conf_history.append(detection.conf)

        # update feature
        if detection.feature is not None:
            self.update_feature(detection.feature)

    def update_feature(self, feature: np.ndarray):
        tmp_feat = feature / np.linalg.norm(feature)
        if self.smooth_feat is None:
            self.smooth_feat = tmp_feat
        else:
            self.smooth_feat = self.ema_alpha * self.smooth_feat + (1. - self.ema_alpha) * tmp_feat
            self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update(self):
        if self.is_matched:
            self.hits += 1
            self.time_since_update = 0
            self.x, self.x_cov = self.kf.update(self.width, self.height)
            self.width, self.height = self.state2wh()

            if self.is_tentative() and (self.hits >= self.init_age or self.conf >= self.track_confirm_thr):
                self.track_state = TrackState.Confirmed
            elif self.is_rematched():
                self.track_state = TrackState.Confirmed
            elif self.is_lost():
                self.track_state = TrackState.Rematched

        else:
            self.hits = 0
            # self.conf *= self.confidence_decay

            if self.is_tentative():
                self.track_state = TrackState.Ambiguous
            elif self.time_since_update > self.max_age:
                self.track_state = TrackState.Ambiguous
            elif self.is_confirmed():
                self.track_state = TrackState.Lost

            if not self.is_valid_bbox():
                self.track_state = TrackState.Ambiguous

    def is_tentative(self):
        return self.track_state == TrackState.Tentative

    def is_confirmed(self):
        return self.track_state == TrackState.Confirmed or self.track_state == TrackState.Rematched

    def is_ambiguous(self):
        return self.track_state == TrackState.Ambiguous

    def is_rematched(self):
        return self.track_state == TrackState.Rematched

    def is_lost(self):
        return self.track_state == TrackState.Lost

    def is_valid_bbox(self):
        aspect_ratio = self.width / self.height
        area = self.width * self.height
        if self.only_pedestrian:
            return aspect_ratio <= self.aspect_ratio_thr and area >= self.area_thr
        else:
            return area >= self.area_thr

    def get_xyxy(self, padding_ratio=1.0):
        if padding_ratio == 1.0:
            padded_xyxy = [
                self.x[0] - self.width / 2,
                self.x[1] - self.height / 2,
                self.x[0] + self.width / 2,
                self.x[1] + self.height / 2
            ]
        else:
            padded_xyxy = [
                self.x[0] - self.width * padding_ratio / 2,
                self.x[1] - self.height * padding_ratio / 2,
                self.x[0] + self.width * padding_ratio / 2,
                self.x[1] + self.height * padding_ratio / 2
            ]
        return np.array(padded_xyxy)[..., 0]
