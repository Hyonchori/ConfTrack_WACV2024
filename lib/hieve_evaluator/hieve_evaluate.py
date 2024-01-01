import os
from pathlib import Path

import natsort
import motmetrics as mm

from hieve_evaluator import Evaluator


def eval_mota(data_root, pred_root):
    print(f'\n=== Processing {pred_root}... \n')
    accs = []
    seqs = [s for s in natsort.natsorted(os.listdir(data_root))]
    for seq in seqs:
        pred_path = os.path.join(pred_root, seq)
        evaluator = Evaluator(data_root, seq, 'mot')
        accs.append(evaluator.eval_file(pred_path))
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    return strsummary


if __name__ == '__main__':
    data_root = '/home/jhc/Desktop/dataset/open_dataset/HiEve/HIE20labels/labels/train/track1'

    ''' ConfTrack '''
    # pred_root = '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/tracker/ConfTrack_hieve/HiEve_train/all/HiEve-train/ConfTrack_hieve/data'

    ''' SORT '''
    # pred_root = '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/SORT/HiEve_train/sort/HiEve-train/reference_tracker_custom/data'

    ''' DeepSORT '''
    # pred_root = '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/DeepSORT/HiEve_train/deepsort/HiEve-train/reference_tracker_custom/data'
    #
    ''' BYTETrack '''
    # pred_root = '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/BYTE/HiEve_train/byte/HiEve-train/reference_tracker_custom/data'
    #
    ''' OCSORT '''
    pred_root = '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/OC-SORT/HiEve_train/ocsort/HiEve-train/reference_tracker_custom/data'
    #
    ''' DeepOCSORT '''
    # pred_root = '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/DeepOCSORT/HiEve_train/depocsort/HiEve-train/reference_tracker_custom/data'
    #
    ''' BoTSORT '''
    # pred_root = '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/BoTSORT/HiEve_train/botsort/HiEve-train/reference_tracker_custom/data'

    save_path = os.path.join(Path(pred_root).parents[0], 'result.txt')

    strsummary = eval_mota(data_root, pred_root)

    print('\n' + strsummary)
    with open(save_path, 'w') as f:
        f.write(strsummary)
