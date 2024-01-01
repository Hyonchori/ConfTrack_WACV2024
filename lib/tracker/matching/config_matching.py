from .matching_strategies import \
    associate_conftrack, \
    associate_byte, \
    associate_conftrack_kitti


def get_matching_fn(cfg):
    matching_dict = {
        'conftrack': associate_conftrack,
        'byte': associate_byte,
        'conftrack_kitti': associate_conftrack_kitti
    }
    if cfg.type_matching not in matching_dict:
        raise KeyError(f'Given type_matching "{cfg.type_matching}" is not in {matching_dict}')

    return matching_dict[cfg.type_matching]
