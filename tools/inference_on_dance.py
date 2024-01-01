# Make inference on DanceTrack dataset

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

from lib.datasets.DanceTrack.dancetrack_dataset import get_dance_videos

from lib.detector.config_detector import get_detector
from lib.detector.detector_utils import scale_coords, clip_coords

from lib.tracker.detection.config_detection import get_detections
from lib.tracker.base_tracker import BaseTracker


@torch.no_grad()
def main(args):
    # Arguments for DanceTrack dataset
    dance_root = args.dance_root
    target_split = args.target_split
    target_vid = args.target_vid
    if target_vid is not None and not isinstance(target_vid, list):
        target_vid = [target_vid]

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

    # load DanceTrack dataset
    dance_videos = get_dance_videos(
        dance_root=dance_root,
        target_split=target_split,
        target_vid=target_vid,
        cfg=trk_cfg,
        input_size=detector.input_size if detector.model is not None else view_size
    )

    # make save directory
    det_out_dir = f'{FILE.parents[1]}/results/detector/{trk_cfg.type_detector}/DanceTrack_{target_split}'
    det_save_dir = increment_path(Path(det_out_dir) / trk_cfg.detector_result_dir, exist_ok=False)
    if save_det:
        det_save_dir.mkdir(parents=True, exist_ok=True)
        trk_cfg.save_opt(det_save_dir)
        print(f"\nSave directory '{det_save_dir}' is created!")

    trk_out_dir = f'{FILE.parents[1]}/results/tracker/{trk_cfg.tracker_name}/DanceTrack_{target_split}'
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
    for vid_idx, dance_dataset in enumerate(dance_videos):
        vid_name = dance_dataset.dataset.dataset_name
        print(f"\n--- Processing {vid_idx + 1} / {len(dance_videos)}'s video: {vid_name}")

        # initialize tracker's track list and track id
        tracker.initialize(vid_name, target_split)

        # create iterator on current video
        time.sleep(0.5)
        iterator = tqdm(enumerate(dance_dataset), total=len(dance_dataset),
                        desc=f'{vid_idx + 1}/{len(dance_videos)}: {vid_name}') \
            if not not_vis_progress_bar else enumerate(dance_dataset)

        if save_vid:
            save_vid_path = os.path.join(trk_save_dir, f'{vid_name}.mp4')
            vid_writer = cv2.VideoWriter(save_vid_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         save_vid_fps, (view_size[1], view_size[0]))

        dance_det_pred = ''
        dance_trk_pred = ''
        ts_load = time.time()
        for i, iter_data in iterator:
            if dance_dataset.dataset.use_detector and dance_dataset.dataset.use_extractor:
                img_raw, img, img_ori_size, det = iter_data

            elif dance_dataset.dataset.use_detector and not dance_dataset.dataset.use_extractor:
                img_raw, img, det = iter_data

            elif not dance_dataset.dataset.use_detector and dance_dataset.dataset.use_extractor:
                img_raw, img_ori_size, det = iter_data

            else:
                img_raw, det = iter_data
            img_v = img_raw[0]
            te_load = time.time()

            if not_vis_progress_bar:
                print(f'\n--- {vid_name}: {i + 1} / {len(dance_dataset)}')

            # make detection
            ts_det = time.time()
            if dance_dataset.dataset.use_detector:
                img = img.to(device).type_as(next(detector.model.parameters()))
                det = detector(img)[0]
                if det is not None:
                    det = det.cpu().numpy()
                    scale_coords(detector.input_size, det, img_raw[0].shape[:2], center_pad=False)
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
            if dance_dataset.dataset.use_extractor:
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
                            dance_det_pred += f'{i + 1},-1,{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[4] * bbox[5]}\n'
                        else:
                            dance_det_pred += f'{i + 1},-1,{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[4]}\n'

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
                    trk_xyxy[[0, 2]] = trk_xyxy[[0, 2]].clip(0, img_raw[0].shape[1])  # x1, x2
                    trk_xyxy[[1, 3]] = trk_xyxy[[1, 3]].clip(0, img_raw[0].shape[0])  # y1, y2
                    trk_width = trk_xyxy[2] - trk_xyxy[0]
                    trk_height = trk_xyxy[3] - trk_xyxy[1]
                    dance_trk_pred += f'{i + 1},{trk_id},' + \
                                      f'{trk_xyxy[0]},{trk_xyxy[1]},{trk_width},{trk_height},1,-1,-1,-1\n'

            # visualize detection and tracking
            ts_vis = time.time()
            if not not_vis or save_vid:
                if not not_vis_det:
                    plot_detection(img_v, detections, {0: 'person'}, hide_cls=True, hide_confidence=False)

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

                plot_info(img_v, f'{vid_name}: {i + 1} / {len(dance_dataset)}',
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
            pred_save_path = os.path.join(det_save_dir, f'{vid_name}.txt')
            with open(pred_save_path, 'w') as f:
                f.write(dance_det_pred)

        # save tracking prediction for TrackEval
        if save_trk:
            track_save_dir = os.path.join(trk_save_dir, f'DanceTrack-{target_split}', trk_cfg.tracker_name,
                                          'data')  # for using TrackEval code
            if not os.path.isdir(track_save_dir):
                os.makedirs(track_save_dir)

            pred_path = os.path.join(track_save_dir, f'{vid_name}.txt')
            with open(pred_path, 'w') as f:
                f.write(dance_trk_pred)
                # print(f'\ttrack prediction result is saved in "{pred_path}"!')
            time.sleep(0.05)

    if save_det:
        print(f'\ndetection prediction results are saved in "{det_save_dir}"!')
    if save_trk:
        print(f'\ntrack prediction results are saved in "{trk_save_dir}"!')

    end_time = time.time()
    print(f'\nTotal elapsed time: {end_time - start_time:.2f}')


def get_args():
    parser = argparse.ArgumentParser()

    # Arguments for DanceTrack dataset
    dance_root = '/home/jhc/Desktop/dataset/open_dataset/DanceTrack'
    parser.add_argument('--dance_root', type=str, default=dance_root)

    target_split = 'val'
    parser.add_argument('--target_split', type=str, default=target_split,
                        help='select in ["train", "val", "test"]')

    target_vid = None
    parser.add_argument('--target_vid', type=str, default=target_vid, nargs='+',
                        help='None: all videos, other numbers: target videos, '
                             'for train, select in [1, 2, 6, 8, 12, 15, 16, 20, 23, 24, 27, 29, 32, 33, 37, 39, 44, '
                             '51, 52, 53, 55, 57, 61, 62, 66, 68, 69, 72, 74, 75, 80, 82, 83, 86, 87, 96, 98, 99], '
                             'for val, select in [4, 5, 7, 10, 14, 18, 19, 25, 26, 30, 34, 35, 41, 43, 47, 58, 63, 65, '
                             '73, 77, 79, 81, 90, 94, 97], '
                             'for test, select in [3, 9, 11, 13, 17, 21, 22, 28, 31, 36, 38, 40, 42, 46, 48, 50, 54, '
                             '56, 59, 60, 64, 67, 70, 71, 76, 78, 84, 85, 88, 89, 91, 92, 93, 95, 100]')

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
