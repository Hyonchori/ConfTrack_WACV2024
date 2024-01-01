from .fast_reid.fast_reid_wrapper import get_wrapped_fast_reid


def get_extractor(cfg, device=None):
    extractor_dict = {
        'fast_reid': get_wrapped_fast_reid
    }
    if cfg.type_extractor not in extractor_dict:
        raise ValueError(f'type_extractor should be one of {extractor_dict.keys()}, but given {cfg.type_extractor}')

    return extractor_dict[cfg.type_extractor](cfg, device)
