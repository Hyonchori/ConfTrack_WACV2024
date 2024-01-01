from typing import List

import lap
import numpy as np

from ..track.base_track import BaseTrack, TrackState
from ..detection.base_detection import BaseDetection
from ..cost.cost_matrices import get_iou_cost, get_embedding_cost, get_confidence_fused_cost


def linear_assignment(cost_mat, row_indices, col_indices, matching_thresh: float = 0.8, gate_mat=None):
    if cost_mat.size == 0:
        return [], row_indices, col_indices
    matches = []
    if gate_mat is not None:
        cost_mat[gate_mat == 0] = np.inf
    cost, x, y = lap.lapjv(cost_mat, extend_cost=True, cost_limit=matching_thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.asarray(row_indices)[np.where(x < 0)[0]]
    unmatched_b = np.asarray(col_indices)[np.where(y < 0)[0]]
    matches = [[row_indices[pair[0]], col_indices[pair[1]]] for pair in matches]
    return matches, unmatched_a.tolist(), unmatched_b.tolist()


def associate_conftrack(
        cfg,
        trk_list: List[BaseTrack],
        det_list: List[BaseDetection],
        img_for_extractor: np.ndarray = None,
        extractor=None
):
    unmatched_trk_indices = list(range(len(trk_list)))
    unmatched_det_indices = list(range(len(det_list)))

    if len(det_list) == 0:
        return [], unmatched_trk_indices, unmatched_det_indices

    det_high_indices = [i for i in unmatched_det_indices
                        if det_list[i].conf >= cfg.detection_high_thr]
    det_low_indices = [i for i in unmatched_det_indices
                       if cfg.detection_low_thr <= det_list[i].conf < cfg.detection_high_thr]
    det_xyxys = [det_list[i].xyxy for i in det_high_indices]

    if det_xyxys and extractor is not None:
        det_feats = extractor(det_xyxys, img_for_extractor)
        for i, det_feat in zip(det_high_indices, det_feats):
            det_list[i].feature = det_feat

    trk_conf_indices = [i for i in unmatched_trk_indices
                        if trk_list[i].is_lost() or trk_list[i].is_confirmed() or
                        (trk_list[i].is_tentative() and trk_list[i].conf >= cfg.track_confirm_thr)]
    trk_tent_indices = [i for i in unmatched_trk_indices
                        if trk_list[i].is_tentative() and trk_list[i].conf < cfg.track_confirm_thr]

    ''' First matching: high_confident_track <-> high_confident_detection (from BoTSORT) '''
    iou_cost_first, iou_gate_first = get_iou_cost(trk_list, det_list, trk_conf_indices, det_high_indices,
                                                  iou_dist_thresh=cfg.first_matching_iou_thresh)
    if extractor is not None:
        emb_cost, emb_gate = get_embedding_cost(trk_list, det_list, trk_conf_indices, det_high_indices,
                                                embedding_dist_thresh=cfg.first_matching_emb_thresh)

    if cfg.use_CFCM:
        iou_cost_first, _ = get_confidence_fused_cost(iou_cost_first, det_list, det_high_indices)
        if extractor is not None:
            emb_cost, _ = get_confidence_fused_cost(emb_cost, det_list, det_high_indices)

    if extractor is not None:
        emb_cost[emb_gate == 0] = 1.0
        emb_cost[iou_gate_first == 0] = 1.0
        cost_mat = np.minimum(iou_cost_first, emb_cost)  # cost matrix from BoTSORT
    else:
        cost_mat = iou_cost_first

    matches_first, unmatched_trk_conf_indices, unmatched_det_high_indices = linear_assignment(
        cost_mat, trk_conf_indices, det_high_indices,
        matching_thresh=cfg.first_matching_thresh, gate_mat=iou_gate_first
    )

    ''' Second matching: unmatched_high_confident_track <-> low_confident_detection (from BYTETrack) '''
    r_trk_conf_indices = [i for i in unmatched_trk_conf_indices if
                          trk_list[i].is_confirmed() or trk_list[i].is_lost()]

    iou_cost_second, iou_gate_second = get_iou_cost(trk_list, det_list, r_trk_conf_indices, det_low_indices,
                                                    iou_dist_thresh=cfg.second_matching_iou_thresh)
    # if cfg.use_CFCM:
    #     iou_cost_second, _ = get_confidence_fused_cost(iou_cost_second, det_list, det_low_indices)
    matches_second, unmatched_r_trk_conf_indices, unmatched_det_low_indices = linear_assignment(
        iou_cost_second, r_trk_conf_indices, det_low_indices,
        matching_thresh=cfg.second_matching_iou_thresh, gate_mat=iou_gate_second
    )

    if cfg.use_LCTM:
        ''' Low confidence matching: low_confident_track <-> unmatched_high_confident_detection (LCTM) '''
        iou_cost_tent_conf, _ = get_iou_cost(trk_list, trk_list,
                                             trk_tent_indices, unmatched_r_trk_conf_indices)
        matches_tent_conf, unmatched_trk_tent_indices, unmatched_r_trk_conf_indices = linear_assignment(
            iou_cost_tent_conf, trk_tent_indices, unmatched_r_trk_conf_indices,
            matching_thresh=cfg.low_confidence_matching_negli_thresh, gate_mat=None
        )

        iou_cost_tent_high, _ = get_iou_cost(trk_list, det_list,
                                             unmatched_trk_tent_indices, unmatched_det_high_indices)
        # if cfg.use_CFCM:
        #     iou_cost_tent_high, _ = get_confidence_fused_cost(iou_cost_tent_high, det_list, unmatched_det_high_indices)
        matches_tent_high, _, _ = linear_assignment(
            iou_cost_tent_high, unmatched_trk_tent_indices, unmatched_det_high_indices,
            matching_thresh=cfg.low_confidence_matching_thresh, gate_mat=None
        )
    else:
        matches_tent_high = []

    matches = matches_first + matches_second + matches_tent_high
    unmatched_trk_indices = list(set(unmatched_trk_indices) - set([x[0] for x in matches]))
    unmatched_det_indices = list(set(unmatched_det_indices) - set([x[1] for x in matches]))

    return matches, unmatched_trk_indices, unmatched_det_indices


def associate_iou(
        trk_list: List[BaseTrack],
        det_list: List[BaseDetection],
        iou_dist_thresh: float = 0.5
):
    unmatched_trk_indices = list(range(len(trk_list)))
    unmatched_det_indices = list(range(len(det_list)))

    if len(det_list) == 0:
        return [], unmatched_trk_indices, unmatched_det_indices

    iou_cost, iou_gate = get_iou_cost(trk_list, det_list, unmatched_trk_indices, unmatched_det_indices,
                                      iou_dist_thresh=iou_dist_thresh)
    matches, unmatched_trk_indices, unmatched_det_indices = linear_assignment(
        iou_cost, unmatched_trk_indices, unmatched_det_indices,
        matching_thresh=iou_dist_thresh, gate_mat=iou_gate
    )

    return matches, unmatched_trk_indices, unmatched_det_indices, iou_cost


def associate_byte(
        cfg,
        trk_list: List[BaseTrack],
        det_list: List[BaseDetection],
        img_for_extractor: np.ndarray = None,
        extractor=None
):
    unmatched_trk_indices = list(range(len(trk_list)))
    unmatched_det_indices = list(range(len(det_list)))

    if len(det_list) == 0:
        return [], unmatched_trk_indices, unmatched_det_indices

    det_high_indices = [i for i in unmatched_det_indices
                        if det_list[i].conf >= cfg.detection_high_thr]
    det_low_indices = [i for i in unmatched_det_indices
                       if cfg.detection_low_thr <= det_list[i].conf < cfg.detection_high_thr]
    det_xyxys = [det_list[i].xyxy for i in det_high_indices]

    if det_xyxys and extractor is not None:
        det_feats = extractor(det_xyxys, img_for_extractor)
        for i, det_feat in zip(det_high_indices, det_feats):
            det_list[i].feature = det_feat

    trk_conf_indices = [i for i in unmatched_trk_indices
                        if trk_list[i].is_lost() or trk_list[i].is_confirmed() or
                        (trk_list[i].is_tentative() and trk_list[i].conf >= cfg.track_confirm_thr)]
    trk_tent_indices = [i for i in unmatched_trk_indices
                        if trk_list[i].is_tentative() and trk_list[i].conf < cfg.track_confirm_thr]

    ''' First matching: high_confident_track <-> high_confident_detection (from BoTSORT) '''
    iou_cost_first, iou_gate_first = get_iou_cost(trk_list, det_list, trk_conf_indices, det_high_indices,
                                                  iou_dist_thresh=cfg.first_matching_iou_thresh)
    if extractor is not None:
        emb_cost, emb_gate = get_embedding_cost(trk_list, det_list, trk_conf_indices, det_high_indices,
                                                embedding_dist_thresh=cfg.first_matching_emb_thresh)

    iou_cost_first, _ = get_confidence_fused_cost(iou_cost_first, det_list, det_high_indices)
    if cfg.use_CFCM:
        if extractor is not None:
            emb_cost, _ = get_confidence_fused_cost(emb_cost, det_list, det_high_indices)

    if extractor is not None:
        emb_cost[emb_gate == 0] = 1.0
        emb_cost[iou_gate_first == 0] = 1.0
        cost_mat = np.minimum(iou_cost_first, emb_cost)  # cost matrix from BoTSORT
    else:
        cost_mat = iou_cost_first

    matches_first, unmatched_trk_conf_indices, unmatched_det_high_indices = linear_assignment(
        cost_mat, trk_conf_indices, det_high_indices,
        matching_thresh=cfg.first_matching_thresh, gate_mat=iou_gate_first
    )

    ''' Second matching: unmatched_high_confident_track <-> low_confident_detection (from BYTETrack) '''
    r_trk_conf_indices = [i for i in unmatched_trk_conf_indices if
                          trk_list[i].is_confirmed() or trk_list[i].is_lost()]

    iou_cost_second, iou_gate_second = get_iou_cost(trk_list, det_list, r_trk_conf_indices, det_low_indices,
                                                    iou_dist_thresh=cfg.second_matching_iou_thresh)
    # if cfg.use_CFCM:
    #     iou_cost_second, _ = get_confidence_fused_cost(iou_cost_second, det_list, det_low_indices)
    matches_second, unmatched_r_trk_conf_indices, unmatched_det_low_indices = linear_assignment(
        iou_cost_second, r_trk_conf_indices, det_low_indices,
        matching_thresh=cfg.second_matching_iou_thresh, gate_mat=iou_gate_second
    )

    if cfg.use_LCTM:
        ''' Low confidence matching: low_confident_track <-> unmatched_high_confident_detection (LCTM) '''
        iou_cost_tent_conf, _ = get_iou_cost(trk_list, trk_list,
                                             trk_tent_indices, unmatched_r_trk_conf_indices)
        matches_tent_conf, unmatched_trk_tent_indices, unmatched_r_trk_conf_indices = linear_assignment(
            iou_cost_tent_conf, trk_tent_indices, unmatched_r_trk_conf_indices,
            matching_thresh=cfg.low_confidence_matching_negli_thresh, gate_mat=None
        )

        iou_cost_tent_high, _ = get_iou_cost(trk_list, det_list,
                                             unmatched_trk_tent_indices, unmatched_det_high_indices)
        # if cfg.use_CFCM:
        #     iou_cost_tent_high, _ = get_confidence_fused_cost(iou_cost_tent_high, det_list, unmatched_det_high_indices)
        matches_tent_high, _, _ = linear_assignment(
            iou_cost_tent_high, unmatched_trk_tent_indices, unmatched_det_high_indices,
            matching_thresh=cfg.low_confidence_matching_thresh, gate_mat=None
        )
    else:
        matches_tent_high = []

    matches = matches_first + matches_second + matches_tent_high
    unmatched_trk_indices = list(set(unmatched_trk_indices) - set([x[0] for x in matches]))
    unmatched_det_indices = list(set(unmatched_det_indices) - set([x[1] for x in matches]))

    return matches, unmatched_trk_indices, unmatched_det_indices


KITTI_CLASSES = {1: 'Car', 2: 'Car', 3: 'Car',
                 4: 'Person', 5: 'Person', 6: 'Person', 7: 'Person',
                 8: 'Tram', 9: 'Misc', 10: 'DontCare'}


def associate_conftrack_kitti(
        cfg,
        trk_list: List[BaseTrack],
        det_list: List[BaseDetection],
        img_for_extractor: np.ndarray = None,
        extractor=None
):
    unmatched_trk_indices = list(range(len(trk_list)))
    unmatched_det_indices = list(range(len(det_list)))

    if len(det_list) == 0:
        return [], unmatched_trk_indices, unmatched_det_indices

    det_high_indices = [i for i in unmatched_det_indices
                        if det_list[i].conf >= cfg.detection_high_thr]
    det_low_indices = [i for i in unmatched_det_indices
                       if cfg.detection_low_thr <= det_list[i].conf < cfg.detection_high_thr]
    det_xyxys = [det_list[i].xyxy for i in det_high_indices]

    if det_xyxys and extractor is not None:
        det_feats = extractor(det_xyxys, img_for_extractor)
        for i, det_feat in zip(det_high_indices, det_feats):
            det_list[i].feature = det_feat

    trk_conf_indices = [i for i in unmatched_trk_indices
                        if trk_list[i].is_lost() or trk_list[i].is_confirmed() or
                        (trk_list[i].is_tentative() and trk_list[i].conf >= cfg.track_confirm_thr)]
    trk_tent_indices = [i for i in unmatched_trk_indices
                        if trk_list[i].is_tentative() and trk_list[i].conf < cfg.track_confirm_thr]

    ''' First matching: high_confident_track <-> high_confident_detection (from BoTSORT) '''
    iou_cost_first, iou_gate_first = get_iou_cost(trk_list, det_list, trk_conf_indices, det_high_indices,
                                                  iou_dist_thresh=cfg.first_matching_iou_thresh)
    if extractor is not None:
        emb_cost, emb_gate = get_embedding_cost(trk_list, det_list, trk_conf_indices, det_high_indices,
                                                embedding_dist_thresh=cfg.first_matching_emb_thresh)
        ''' Only use embedding cost for person class '''
        # for i, trk_idx in enumerate(trk_conf_indices):
        #     if trk_list[trk_idx].cls not in [4, 5, 6, 7]:
        #         emb_cost[i, :] = 1.0
        # for j, det_idx in enumerate(det_high_indices):
        #     if det_list[det_idx].cls not in [4, 5, 6, 7]:
        #         emb_cost[:, j] = 1.0

    if cfg.use_CFCM:
        iou_cost_first, _ = get_confidence_fused_cost(iou_cost_first, det_list, det_high_indices)
        if extractor is not None:
            emb_cost, _ = get_confidence_fused_cost(emb_cost, det_list, det_high_indices)

    if extractor is not None:
        emb_cost[emb_gate == 0] = 1.0
        emb_cost[iou_gate_first == 0] = 1.0
        cost_mat = np.minimum(iou_cost_first, emb_cost)  # cost matrix from BoTSORT
    else:
        cost_mat = iou_cost_first

    ''' Make unmatching between other classes '''
    for i, trk_idx in enumerate(trk_conf_indices):
        for j, det_idx in enumerate(det_high_indices):
            if KITTI_CLASSES[trk_list[trk_idx].cls] != KITTI_CLASSES[det_list[det_idx].cls]:
                cost_mat[i, j] = 1.0

    matches_first, unmatched_trk_conf_indices, unmatched_det_high_indices = linear_assignment(
        cost_mat, trk_conf_indices, det_high_indices,
        matching_thresh=cfg.first_matching_thresh, gate_mat=iou_gate_first
    )

    ''' Second matching: unmatched_high_confident_track <-> low_confident_detection (from BYTETrack) '''
    r_trk_conf_indices = [i for i in unmatched_trk_conf_indices if
                          trk_list[i].is_confirmed() or trk_list[i].is_lost()]

    iou_cost_second, iou_gate_second = get_iou_cost(trk_list, det_list, r_trk_conf_indices, det_low_indices,
                                                    iou_dist_thresh=cfg.second_matching_iou_thresh)
    # if cfg.use_CFCM:
    #     iou_cost_second, _ = get_confidence_fused_cost(iou_cost_second, det_list, det_low_indices)
    ''' Make unmatching between other classes '''
    for i, trk_idx in enumerate(r_trk_conf_indices):
        for j, det_idx in enumerate(det_low_indices):
            if KITTI_CLASSES[trk_list[trk_idx].cls] != KITTI_CLASSES[det_list[det_idx].cls]:
                iou_cost_second[i, j] = 1.0
    matches_second, unmatched_r_trk_conf_indices, unmatched_det_low_indices = linear_assignment(
        iou_cost_second, r_trk_conf_indices, det_low_indices,
        matching_thresh=cfg.second_matching_iou_thresh, gate_mat=iou_gate_second
    )

    if cfg.use_LCTM:
        ''' Low confidence matching: low_confident_track <-> unmatched_high_confident_detection (LCTM) '''
        iou_cost_tent_conf, _ = get_iou_cost(trk_list, trk_list,
                                             trk_tent_indices, unmatched_r_trk_conf_indices)
        matches_tent_conf, unmatched_trk_tent_indices, unmatched_r_trk_conf_indices = linear_assignment(
            iou_cost_tent_conf, trk_tent_indices, unmatched_r_trk_conf_indices,
            matching_thresh=cfg.low_confidence_matching_negli_thresh, gate_mat=None
        )

        iou_cost_tent_high, _ = get_iou_cost(trk_list, det_list,
                                             unmatched_trk_tent_indices, unmatched_det_high_indices)
        # if cfg.use_CFCM:
        #     iou_cost_tent_high, _ = get_confidence_fused_cost(iou_cost_tent_high, det_list, unmatched_det_high_indices)
        ''' Make unmatching between other classes '''
        for i, trk_idx in enumerate(unmatched_trk_tent_indices):
            for j, det_idx in enumerate(unmatched_det_high_indices):
                if KITTI_CLASSES[trk_list[trk_idx].cls] != KITTI_CLASSES[det_list[det_idx].cls]:
                    iou_cost_tent_high[i, j] = 1.0
        matches_tent_high, _, _ = linear_assignment(
            iou_cost_tent_high, unmatched_trk_tent_indices, unmatched_det_high_indices,
            matching_thresh=cfg.low_confidence_matching_thresh, gate_mat=None
        )
    else:
        matches_tent_high = []

    matches = matches_first + matches_second + matches_tent_high
    unmatched_trk_indices = list(set(unmatched_trk_indices) - set([x[0] for x in matches]))
    unmatched_det_indices = list(set(unmatched_det_indices) - set([x[1] for x in matches]))

    return matches, unmatched_trk_indices, unmatched_det_indices
