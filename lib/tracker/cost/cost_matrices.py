from typing import List

import numpy as np
from scipy.spatial.distance import cdist
from cython_bbox import bbox_overlaps as bbox_ious

from ..track.base_track import BaseTrack
from ..detection.base_detection import BaseDetection


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def get_iou_cost(
        tracks: List[BaseTrack],
        detections: List[BaseDetection],
        trk_indices: List[int] = None,
        det_indices: List[int] = None,
        padding_ratio: float = 1.0,
        iou_dist_thresh: float = 1 - 0.3
):
    if trk_indices is None:
        trk_indices = np.arange(len(tracks))
    if det_indices is None:
        det_indices = np.arange(len(detections))

    trk_xyxys = [tracks[i].get_xyxy(padding_ratio=padding_ratio) for i in trk_indices]
    det_xyxys = [detections[i].get_xyxy(padding_ratio=padding_ratio) for i in det_indices]

    iou_mat = ious(trk_xyxys, det_xyxys)
    iou_cost = 1 - iou_mat

    iou_gate = np.zeros_like(iou_cost)
    iou_gate[iou_cost <= iou_dist_thresh] = 1

    return iou_cost, iou_gate


def get_embedding_cost(
        tracks: List[BaseTrack],
        detections: List[BaseDetection],
        trk_indices: List[int] = None,
        det_indices: List[int] = None,
        embedding_dist_thresh: float = 0.25
):
    if trk_indices is None:
        trk_indices = np.arange(len(tracks))
    if det_indices is None:
        det_indices = np.arange(len(detections))

    embedding_cost = np.zeros((len(trk_indices), len(det_indices)), dtype=np.float32)
    embedding_gate = np.zeros_like(embedding_cost)
    if embedding_cost.size == 0:
        return embedding_cost, embedding_gate

    det_feats = np.asarray([detections[i].feature for i in det_indices], dtype=np.float)
    trk_feats = np.asarray([tracks[i].smooth_feat for i in trk_indices], dtype=np.float) if len(tracks) \
        else np.empty((0, det_feats.shape[1]))

    embedding_cost = np.maximum(0.0, cdist(trk_feats, det_feats, 'cosine')) / 2.0

    embedding_gate[embedding_cost <= embedding_dist_thresh] = 1

    return embedding_cost, embedding_gate


def get_confidence_fused_cost(
        cost: np.ndarray,
        detections: List[BaseDetection],
        det_indices: List[int],
        just_multiply: bool = False
):
    gate_mat = np.ones_like(cost)
    if cost.size == 0:
        return cost, gate_mat

    confs = np.array([detections[col].conf for col in det_indices])
    confs = np.expand_dims(confs, axis=0).repeat(cost.shape[0], axis=0)

    if just_multiply:
        fused_cost = cost * confs
    else:
        ''' Confidence Fused Cost Matrix '''
        sim = 1 - cost
        sim *= confs
        fused_cost = 1 - sim
    return fused_cost, gate_mat
