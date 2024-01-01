# Make inference on MOT17/MOT20 dataset

import argparse
import os
import sys
import time
import importlib
import warnings
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm

from lib.custom_utils.torch_utils import select_device
from lib.custom_utils.general_utils import increment_path, xywh2xyxy, xyxy2xywh
from lib.custom_utils.plot_utils import letterbox, plot_info, plot_detection, plot_track

from lib.datasets.MOT.mot_dataset import get_mot_videos, MOT_CLASSES

from lib.detector.config_detector import get_detector
from lib.detector.detector_utils import scale_coords, clip_coords

from lib.tracker.detection.config_detection import get_detections
from lib.tracker.base_tracker import BaseTracker


@torch.no_grad()
def main(args):
    # Arguments for MOT17/MOT20 dataset
    mot_root = args.mot_root
    target_select = args.target_select
    target_split = args.target_split
    target_vid = args.target_vid
    if target_vid is not None and not isinstance(target_vid, list):
        target_vid = [target_vid]
    target_det = args.target_det
    if target_det is not None and not isinstance(target_det, list):
        target_det = [target_det]

    # Arguments for tracker configuration
    trk_cfg_file = args.trk_cfg_file
    sys.path.append((os.path.join(os.path.dirname(FILE.parent), 'cfgs')))
    trk_cfg_file = importlib.import_module(trk_cfg_file)
    trk_cfg = trk_cfg_file.TrackerCFG()

    # General arguments for inference
    device = select_device(args.device)
    not_vis_progress_bar = args.not_vis_progress_bar
    run_name = args.run_name
    not_vis_det = args.not_vis_det
    not_vis_trk = args.not_vis_trk
    not_vis = args.not_vis
    save_det = args.save_det
    save_trk = args.save_trk
    save_vid = args.save_vid
    save_vid_fps = args.save_vid_fps
    view_size = args.view_size

    # load detector using configuration
    detector = get_detector(
        cfg=trk_cfg,
        device=device
    )

    # load tracker using configuration
    tracker = BaseTracker(trk_cfg, device)

    # load MOT dataset
    mot_videos, remain_dets = get_mot_videos(
        mot_root=mot_root,
        target_select=target_select,
        target_split=target_split,
        target_vid=target_vid,
        target_det=target_det,
        cfg=trk_cfg,
        input_size=detector.input_size if detector.model is not None else view_size
    )

    # make save directory
    det_out_dir = f'{FILE.parents[1]}/results/detector/{trk_cfg.type_detector}/{target_select}_{target_split}'
    det_save_dir = increment_path(Path(det_out_dir) / trk_cfg.detector_result_dir, exist_ok=False)
    if save_det:
        det_save_dir.mkdir(parents=True, exist_ok=True)
        trk_cfg.save_opt(det_save_dir)
        print(f"\nSave directory '{det_save_dir}' is created!")

    trk_out_dir = f'{FILE.parents[1]}/results/tracker/{trk_cfg.tracker_name}/{target_select}_{target_split}'
    trk_save_dir = increment_path(Path(trk_out_dir) / run_name, exist_ok=False)
    if save_trk or save_vid:
        trk_save_dir.mkdir(parents=True, exist_ok=True)
        trk_cfg.save_opt(trk_save_dir)
        print(f"\nSave directory '{trk_save_dir}' is created!")

    # pre-inference
    if device.type != "cpu":
        if detector.model is not None:
            detector(torch.zeros(1, 3, *detector.input_size).to(device).
                     type_as(next(detector.model.parameters())))

        if tracker.extractor is not None:
            tracker.extractor(torch.zeros(1, 3, *tracker.extractor.input_size).to(device).
                              type_as(next(tracker.extractor.model.parameters())))

    # iterate videos
    start_time = time.time()
    for vid_idx, mot_dataset in enumerate(mot_videos):
        vid_name = mot_dataset.dataset.dataset_name
        print(f"\n--- Processing {vid_idx + 1} / {len(mot_videos)}'s video: {vid_name}")

        # initialize tracker's track list and track id
        tracker.initialize(vid_name, target_split)

        # create iterator on current video
        time.sleep(0.5)
        iterator = tqdm(enumerate(mot_dataset), total=len(mot_dataset),
                        desc=f'{vid_idx + 1}/{len(mot_videos)}: {vid_name}') \
            if not not_vis_progress_bar else enumerate(mot_dataset)

        if save_vid:
            save_vid_path = os.path.join(trk_save_dir, f'{vid_name}.mp4')
            vid_writer = cv2.VideoWriter(save_vid_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         save_vid_fps, (view_size[1], view_size[0]))

        mot_det_pred = ''
        mot_trk_pred = ''
        ts_load = time.time()
        for i, iter_data in iterator:
            if mot_dataset.dataset.use_detector and mot_dataset.dataset.use_extractor:
                img_raw, img, img_ori_size, det = iter_data

            elif mot_dataset.dataset.use_detector and not mot_dataset.dataset.use_extractor:
                img_raw, img, det = iter_data

            elif not mot_dataset.dataset.use_detector and mot_dataset.dataset.use_extractor:
                img_raw, img_ori_size, det = iter_data

            else:
                img_raw, det = iter_data
            img_v = img_raw[0]
            te_load = time.time()

            if not_vis_progress_bar:
                print(f'\n--- {vid_name}: {i + 1} / {len(mot_dataset)}')

            # make detection
            ts_det = time.time()
            if mot_dataset.dataset.use_detector:
                img = img.to(device).type_as(next(detector.model.parameters()))
                det = detector(img)[0]
                if det is not None:
                    det = det.cpu().numpy()
                    scale_coords(detector.input_size, det, img_raw[0].shape[:2], center_pad=False)
                    if target_select == 'MOT20':
                        clip_coords(det, img_raw[0].shape[:2])
                else:
                    det = np.empty((0, 6))
            else:
                det = det[0]
                if len(det) != 0:
                    det = xywh2xyxy(det)
                else:
                    det = np.empty((0, 6))
            detections = get_detections(trk_cfg, det)
            te_det = time.time()

            # make tracking prediction
            ts_trk = time.time()
            tracker.predict(img_raw[0], detections, img_idx=i)

            # make tracking update
            if mot_dataset.dataset.use_extractor:
                tracker.update(detections, img_for_extractor=img_ori_size)
            else:
                tracker.update(detections)
            te_trk = time.time()

            # write detection results
            if save_det:
                if len(det):
                    bboxes = xyxy2xywh(det)
                    for bbox in bboxes:
                        if bbox[-1] != 0:  # only save pedestrian
                            continue
                        if len(bbox) == 7:
                            mot_det_pred += f'{i + 1},-1,{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[4] * bbox[5]}\n'
                        else:
                            mot_det_pred += f'{i + 1},-1,{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[4]}\n'

            # write tracking results
            if save_trk:
                for track in tracker.tracks:
                    if i == 0:
                        if track.conf < trk_cfg.detection_high_thr:
                            continue
                    else:  # i >= 1:
                        if trk_cfg.save_only_matched and not track.is_matched:
                            continue
                        if trk_cfg.save_only_confirmed and not track.is_confirmed():
                            continue
                    trk_id = track.track_id
                    trk_xyxy = track.get_xyxy()
                    if target_select == 'MOT20':
                        trk_xyxy[[0, 2]] = trk_xyxy[[0, 2]].clip(0, img_raw[0].shape[1])  # x1, x2
                        trk_xyxy[[1, 3]] = trk_xyxy[[1, 3]].clip(0, img_raw[0].shape[0])  # y1, y2
                    trk_width = trk_xyxy[2] - trk_xyxy[0]
                    trk_height = trk_xyxy[3] - trk_xyxy[1]
                    mot_trk_pred += f'{i + 1},{trk_id},' + \
                                    f'{trk_xyxy[0]},{trk_xyxy[1]},{trk_width},{trk_height},1,-1,-1,-1\n'

            # visualize detection and tracking
            ts_vis = time.time()
            if not not_vis or save_vid:
                if not not_vis_det:
                    plot_detection(img_v, detections, MOT_CLASSES, hide_cls=True, hide_confidence=False)

                if not not_vis_trk:
                    img_v = plot_track(
                        img_v, tracker.tracks,
                        bbox_thickness=3,
                        font_size=0.8,
                        font_thickness=2,
                        vis_only_matched=True,
                        vis_only_confirmed=True,
                        target_states=None,
                        target_ids=None,
                    )

                plot_info(img_v, f'{vid_name}: {i + 1} / {len(mot_dataset)}',
                          font_size=2, font_thickness=2)

                if view_size is not None:
                    img_v = letterbox(img_v, view_size, auto=False)[0]

                if not not_vis:
                    cv2.imshow(vid_name, img_v)
                    keyboard_input = cv2.waitKey(1) & 0xff
                    if keyboard_input == ord('q'):
                        break
                    elif keyboard_input == 27:  # 27: esc
                        sys.exit()
            if save_vid:
                vid_writer.write(img_v)
            te_vis = time.time()

            if not_vis_progress_bar:
                load_time = te_load - ts_load
                det_time = te_det - ts_det
                trk_time = te_trk - ts_trk
                vis_time = te_vis - ts_vis
                iter_time = te_vis - te_load
                print(f'load_time: {load_time:.4f}')
                print(f'det_time: {det_time:.4f}')
                print(f'trk_time: {trk_time:.4f}')
                print(f'vis_time: {vis_time:.4f}')
                print(f'\titer_total_time: {iter_time:.4f}')
            ts_load = time.time()

        if not not_vis:
            cv2.destroyWindow(vid_name)

        if save_det:
            if target_select == 'MOT17':
                pred_save_path = os.path.join(det_save_dir,
                                              '-'.join(mot_dataset.dataset.dataset_name.split('-')[:-1]) + '.txt')
            else:  # target_select == 'MOT20'
                pred_save_path = os.path.join(det_save_dir,
                                              mot_dataset.dataset.dataset_name.split('.')[0] + '.txt')
            with open(pred_save_path, 'w') as f:
                f.write(mot_det_pred)

        # save tracking prediction for TrackEval
        if save_trk:
            track_save_dir = os.path.join(trk_save_dir, f'{target_select}-{target_split}', trk_cfg.tracker_name,
                                          'data')  # for using TrackEval code
            if not os.path.isdir(track_save_dir):
                os.makedirs(track_save_dir)

            pred_path = os.path.join(track_save_dir, f'{vid_name}.txt')
            with open(pred_path, 'w') as f:
                f.write(mot_trk_pred)
                # print(f'\ttrack prediction result is saved in "{pred_path}"!')

            if target_select == 'MOT17':
                for remain_det in remain_dets:
                    # for additional prediction file for TrackEval when using specific detection
                    remain_vid_name = '-'.join(vid_name.split('-')[:-1] + [remain_det])
                    remain_path = os.path.join(track_save_dir, f'{remain_vid_name}.txt')
                    with open(remain_path, 'w') as f:
                        f.write(mot_trk_pred)
                        # print(f'\ttrack prediction result is saved in "{remain_path}"!')
            time.sleep(0.05)

    if save_det:
        print(f'\ndetection prediction results are saved in "{det_save_dir}"!')
    if save_trk:
        print(f'\ntrack prediction results are saved in "{trk_save_dir}"!')

    end_time = time.time()
    print(f'\nTotal elapsed time: {end_time - start_time:.2f}')


def get_args():
    parser = argparse.ArgumentParser()

    # Arguments for MOT17/MOT20 dataset
    mot_root = '/home/jhc/Desktop/dataset/open_dataset/mot_test'
    parser.add_argument('--mot_root', type=str, default=mot_root)

    target_select = 'MOT20'
    parser.add_argument('--target_select', type=str, default=target_select,
                        help='select in ["MOT17", "MOT20"]')

    target_split = 'val'
    parser.add_argument('--target_split', type=str, default=target_split,
                        help='select in ["train", "val", "test"]')

    target_vid = None
    parser.add_argument('--target_vid', type=str, default=target_vid, nargs='+',
                        help='None: all videos, other numbers: target videos, '
                             'for MOT17 train/val, select in [2, 4, 5, 9, 10, 11, 13], '
                             'for MOT17 test, select in [1, 3, 6, 7, 8, 12, 14], '
                             'for MOT20 train/val, select in [1, 2, 3, 5], '
                             'for MOT20 test, select in [4, 6, 7, 8]')

    target_det = ['FRCNN']
    parser.add_argument('--target_det', type=str, default=target_det, nargs='+',
                        help='for MOT17, select in ["DPM", "FRCNN", "SDP"]')

    # Arguments for tracker
    trk_cfg_file = 'conftrack_baseline'
    parser.add_argument('--trk_cfg_file', type=str, default=trk_cfg_file,
                        help='# file name of target config in "cfgs" directory')

    # General arguments for inference
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--not_vis_progress_bar', action='store_true')
    parser.add_argument('--run_name', type=str, default='conftrack_inference')
    parser.add_argument('--not_vis_det', action='store_true')
    parser.add_argument('--not_vis_trk', action='store_true')
    parser.add_argument('--not_vis', action='store_true', default=True)
    parser.add_argument('--save_det', action='store_true', default=True)
    parser.add_argument('--save_trk', action='store_true', default=True)
    parser.add_argument('--save_vid', action='store_true', default=True)
    parser.add_argument('--save_vid_fps', type=int, default=20)
    parser.add_argument('--view_size', type=int, nargs='+', default=[720, 1280],
                        help='[height, width]')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    FILE = Path(__file__).absolute()
    warnings.filterwarnings("ignore")
    np.set_printoptions(linewidth=np.inf)
    opt = get_args()
    main(opt)
