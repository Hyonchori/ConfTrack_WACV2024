import numpy as np

from .base_kalmanfilter import BaseKalmanFilter


system_matrix = np.array([
    [1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]
], dtype=np.float32)

projection_matrix = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0]
], dtype=np.float32)


def get_kalmanfilter(cfg, init_measure, init_conf, init_cls):
    return BaseKalmanFilter(
        system_matrix=system_matrix,
        projection_matrix=projection_matrix,
        init_measure=init_measure,
        init_confidence=init_conf,
        init_cls=init_cls,
        std_weight_position=cfg.std_weight_position,
        std_weight_velocity=cfg.std_weight_velocity,
        use_NSAK=cfg.use_NSAK,
        nsa_amplify_factor=cfg.nsa_amplify_factor,
    )
