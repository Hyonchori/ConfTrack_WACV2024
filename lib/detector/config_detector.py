from .yolox.yolox_wrapper import get_wrapped_yolox


def get_detector(cfg, device=None):
    detector_dict = {
        'yolox': get_wrapped_yolox,
    }
    if cfg.type_detector not in detector_dict:
        raise ValueError(f'type_detector should be one of {detector_dict.keys()}, but given {cfg.type_detector}')

    return detector_dict[cfg.type_detector](cfg, device)
