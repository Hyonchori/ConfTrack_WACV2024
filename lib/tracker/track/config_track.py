from .base_track import BaseTrack
from ..detection.base_detection import BaseDetection
from ..kalman_filter.base_kalmanfilter import BaseKalmanFilter


def get_track(cfg, track_id: int, detection: BaseDetection, kalman_filter: BaseKalmanFilter):
    return BaseTrack(
        track_id=track_id,
        detection=detection,
        kalman_filter=kalman_filter,
        max_age=cfg.max_age,
        init_age=cfg.init_age,
        ema_alpha=cfg.ema_alpha,
        aspect_ratio_thr=cfg.aspect_ratio_thr,
        area_thr=cfg.area_thr,
        only_pedestrian=cfg.only_pedestrian,
        use_CWKU=cfg.use_CWKU,
        CWKU_thr=cfg.detection_high_thr if not hasattr(cfg, 'cw_threshold') else cfg.cw_threshold,
        use_CPLT=cfg.use_CPLT,
        track_confirm_thr=cfg.track_confirm_thr,
        confidence_decay=cfg.confidence_decay
    )
