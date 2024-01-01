# Visualize tracking prediction of MOT17, MOT20 as tracjectory

import argparse
import os
from collections import defaultdict

import cv2
import numpy as np
from screeninfo import get_monitors


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


def parsing_mot_pred(pred_path):
    with open(pred_path) as f:
        rets = [x.strip('\n').split(',') for x in f.readlines()]
        gt = defaultdict(list)
        for ret in rets:
            gt[int(ret[0]) - 1].append(ret[1:])
    return gt


def parsing_mot_traj(pred_path, target_ids=None):
    with open(pred_path) as f:
        rets = [x.strip('\n').split(',') for x in f.readlines()]
        traj = defaultdict(list)
        for ret in rets:
            if target_ids is not None and int(float(ret[1])) not in target_ids:
                continue
            traj[int(float(ret[1]))].append([ret[0]] + ret[2:6])
    return traj


def main(args):
    mot_root = args.mot_root
    target_select = args.target_select
    target_split = args.target_split
    target_vid = args.target_vid
    target_ids = args.target_ids
    pred_dirs = args.pred_dirs
    view_size = args.view_size
    single_img = args.single_img
    view_gt = args.view_gt
    if target_vid is not None and not isinstance(target_vid, list):
        target_vid = [target_vid]
    if target_ids is not None and not isinstance(target_ids, list):
        target_ids = [target_ids]

    vid_root = os.path.join(mot_root, target_select, target_split)
    if not os.path.isdir(vid_root):
        raise ValueError(f'Given arguments are wrong!: {mot_root}, {target_select}, {target_split}')

    if target_select == 'MOT17':
        total_vid_list = os.listdir(vid_root)
        if target_vid is not None:
            vid_list = [f'{target_select}-{idx:02d}' for idx in target_vid]
        else:
            vid_list = sorted(list(set(['-'.join(x.split('-')[:-1]) for x in total_vid_list])))
        vid_list = [f'{x}-FRCNN' for x in vid_list if f'{x}-FRCNN' in total_vid_list]
    else:
        if target_vid is not None:
            vid_list = [x for x in os.listdir(vid_root) if int(x.split('-')[-1]) in target_vid]
        else:
            vid_list = sorted(os.listdir(vid_root))

    pred_list = [{x.split('.')[0]: x for x in os.listdir(pred_dir)} for pred_dir in pred_dirs.values()]

    monitors = get_monitors()
    if len(monitors) == 1:
        tmp_monitor = monitors[0]
    else:
        tmp_monitor = monitors[1]
    ms = int(tmp_monitor.height * 0.8), int(tmp_monitor.width * 0.8)
    view_size = view_size if view_size is not None else ms

    for vid_idx, vid_name in enumerate(vid_list):
        print(f"\n--- Processing {vid_idx + 1} / {len(vid_list)}'s video: {vid_name}")
        vid_dir = os.path.join(vid_root, vid_name)
        img_root = os.path.join(vid_dir, 'img1')
        imgs = sorted(os.listdir(img_root))

        if not view_gt:
            pred_files = [x[vid_name] for x in pred_list]
            pred_paths = [os.path.join(pred_dir, pred_file) for pred_dir, pred_file in zip(pred_dirs.values(), pred_files)]
            preds = [parsing_mot_pred(pred_path) for pred_path in pred_paths]

            trajs = [parsing_mot_traj(pred_path, target_ids) for pred_path in pred_paths]
        else:
            gt_path = os.path.join(vid_dir, 'gt', 'gt.txt')
            print(gt_path)
            trajs = [parsing_mot_traj(gt_path, target_ids)]

        img_idx = 0
        total_len = len(imgs)
        start_img_path = os.path.join(img_root, imgs[img_idx])
        start_img = cv2.imread(start_img_path)

        ref_img = np.zeros_like(start_img)

        if not single_img:
            pred_imgs = []
            for i, (tracker_name, traj) in enumerate(zip(pred_dirs.keys(), trajs)):
                tmp_img = ref_img.copy()
                # plot_info(tmp_img, f'{tracker_name}: {vid_name}', font_size=2, font_thickness=2)

                for person_id, person_traj in traj.items():
                    trk_color = colors(person_id, True)
                    for j, spot in enumerate(person_traj):
                        xywh = np.array(list(map(float, spot[1: 5])))
                        pt1, pt2, pt3, pt4 = xywh2pts(xywh)
                        # cv2.circle(tmp_img, pt1, 10, trk_color, -1)
                        # cv2.circle(tmp_img, pt2, 10, trk_color, -1)
                        # cv2.circle(tmp_img, pt3, 10, trk_color, -1)
                        # cv2.circle(tmp_img, pt4, 10, trk_color, -1)
                        if j == 0:
                            xyxy = xywh2xyxy(xywh)
                            cv2.rectangle(tmp_img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                                          [255, 255, 255], 10)
                        elif j == (len(person_traj) - 1):
                            xyxy = xywh2xyxy(xywh)
                            cv2.rectangle(tmp_img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                                          [255, 255, 0], 10)
                        if j != 0:
                            xywh_prior = np.array(list(map(float, person_traj[j - 1][1: 5])))
                            ptp1, ptp2, ptp3, ptp4 = xywh2pts(xywh_prior)
                            cv2.line(tmp_img, ptp1, pt1, [255, 0, 0], 12)
                            cv2.line(tmp_img, ptp2, pt2, [0, 0, 255], 12)
                            cv2.line(tmp_img, ptp3, pt3, [0, 255, 0], 12)
                            cv2.line(tmp_img, ptp4, pt4, [0, 255, 255], 12)

                pred_imgs.append(tmp_img)

            pred_img = np.vstack(pred_imgs)
            pred_img = letterbox(pred_img, view_size, auto=False)[0]

        else:
            pred_img = ref_img.copy()
            for i, (tracker_name, traj) in enumerate(zip(pred_dirs.keys(), trajs)):
                # plot_info(pred_img, f'{tracker_name}: {vid_name}', font_size=2, font_thickness=2)

                for person_id, person_traj in traj.items():
                    trk_color = colors(person_id + i, True)
                    for j, spot in enumerate(person_traj):
                        xywh = np.array(list(map(float, spot[1: 5])))
                        pt1, pt2, pt3, pt4 = xywh2pts(xywh)
                        cv2.circle(pred_img, pt1, 6, trk_color, -1)
                        cv2.circle(pred_img, pt2, 6, trk_color, -1)
                        cv2.circle(pred_img, pt3, 6, trk_color, -1)
                        cv2.circle(pred_img, pt4, 6, trk_color, -1)

                        if j == 0:
                            xyxy = xywh2xyxy(xywh)
                            cv2.rectangle(pred_img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                                          [255, 255, 255], 5)
                        elif j == (len(person_traj) - 1):
                            xyxy = xywh2xyxy(xywh)
                            cv2.rectangle(pred_img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                                          [255, 255, 0], 5)
                        if j != 0:
                            xywh_prior = np.array(list(map(float, person_traj[j - 1][1: 5])))
                            ptp1, ptp2, ptp3, ptp4 = xywh2pts(xywh_prior)
                            cv2.line(pred_img, ptp1, pt1, trk_color, 4)
                            cv2.line(pred_img, ptp2, pt2, trk_color, 4)
                            cv2.line(pred_img, ptp3, pt3, trk_color, 4)
                            cv2.line(pred_img, ptp4, pt4, trk_color, 4)

            pred_img = letterbox(pred_img, view_size, auto=False)[0]

        cv2.imshow(vid_name, pred_img)
        keyboard_input = cv2.waitKey(0) & 0xff
        if keyboard_input == ord('q'):
            break
        elif keyboard_input == 27:  # 27: esc
            import sys
            sys.exit()

        cv2.destroyWindow(vid_name)


def xywh2pts(xywh):
    pt1 = [int(xywh[0]), int(xywh[1])]
    pt2 = [int(xywh[0] + xywh[2]), int(xywh[1])]
    pt3 = [int(xywh[0] + xywh[2]), int(xywh[1] + xywh[3])]
    pt4 = [int(xywh[0]), int(xywh[1] + xywh[3])]
    return pt1, pt2, pt3, pt4


def xywh2xyxy(xywh):
    xyxy = xywh.copy()
    xyxy[2] = int(xyxy[0] + xyxy[2])
    xyxy[3] = int(xyxy[1] + xyxy[3])
    xyxy[0] = int(xyxy[0])
    xyxy[1] = int(xyxy[1])
    return xyxy


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def plot_info(img, info, font_size=1, font_thickness=1):
    label_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_PLAIN, font_size, font_thickness)[0]
    cv2.rectangle(img, (0, 0), (label_size[0] + 10, label_size[1] * 2), [0, 0, 0], -1)
    cv2.putText(img, info, (5, int(label_size[1] * 1.5))
                , cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)


def plot_label(img, xyxy, label, color, font_size=0.4, font_thickness=1):
    txt_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)[0]
    txt_bk_color = [int(c * 0.7) for c in color]
    cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[0]) + txt_size[0] + 1, int(xyxy[1]) - int(txt_size[1] * 1.5)),
                  txt_bk_color, -1)
    cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - int(txt_size[1] * 0.4)),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness)


def parse_args():
    parser = argparse.ArgumentParser()

    # Arguments for KITTI tracking dataset
    mot_root = '/home/jhc/Desktop/dataset/open_dataset/MOT'
    parser.add_argument('--mot_root', type=str, default=mot_root)

    target_select = 'MOT20'
    parser.add_argument('--target_select', type=str, default=target_select)

    target_split = 'val'  # ['train', 'val', 'test']
    parser.add_argument('--target_split', type=str, default=target_split)

    target_vid = 5
    # None: all videos, other numbers: target videos
    parser.add_argument('--target_vid', type=int, default=target_vid, nargs='+')

    # pred_dirs = {  # MOT17 test
    #     # 'baseline': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/tracker/ConfTrack_baseline/MOT17_test/baseline/MOT17-test/ConfTrack_baseline/data',
    #     'all': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/tracker/ConfTrack_ablation_custom/MOT17_test/all/MOT17-test/ConfTrack_ablation_custom/data'
    # }
    # pred_dirs = {  # MOT17 val
    #     'baseline': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/tracker/ConfTrack_ablation_custom/MOT17_val/baseline/MOT17-val/ConfTrack_ablation_custom/data',
    #     'conftrack': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/tracker/ConfTrack_ablation_custom/MOT17_val/all/MOT17-val/ConfTrack_ablation_custom/data',
    #     'sort': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/SORT/MOT17_val/sort/MOT17-val/reference_tracker_custom/data',
    #     'deepsort': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/DeepSORT/MOT17_val/deepsort/MOT17-val/reference_tracker_custom/data',
    #     'byte': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/BYTE/MOT17_val/byte2/MOT17-val/reference_tracker_custom/data',
    #     'ocsort': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/OC-SORT/MOT17_val/ocsort/MOT17-val/reference_tracker_custom/data',
    #     'deepocsort': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/DeepOCSORT/MOT17_val/deepocsort/MOT17-val/reference_tracker_custom/data'
    #     'botsort': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/BoTSORT/MOT17_val/botsort/MOT17-val/reference_tracker_custom/data'
    # }
    pred_dirs = {  # MOT20 val
        # 'baseline': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/tracker/ConfTrack_ablation_custom/MOT20_val/baseline2/MOT20-val/ConfTrack_ablation_custom/data',
        # 'contrack': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/tracker/ConfTrack_ablation_custom/MOT20_val/all2/MOT20-val/ConfTrack_ablation_custom/data',
        # 'sort': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/SORT/MOT20_val/sort/MOT20-val/reference_tracker_custom/data',
        # 'deepsort': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/DeepOCSORT/MOT20_val/deepocsort/MOT20-val/reference_tracker_custom/data',
        # 'byte': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/BYTE/MOT20_val/byte2/MOT20-val/reference_tracker_custom/data',
        # 'ocsort': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/OC-SORT/MOT20_val/ocsort/MOT20-val/reference_tracker_custom/data',
    #     'deepocsort': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/DeepOCSORT/MOT20_val/deepocsort/MOT20-val/reference_tracker_custom/data'
        'botsort': '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/results/reference_tracker_custom/BoTSORT/MOT20_val/botsort/MOT20-val/reference_tracker_custom/data'
    }
    parser.add_argument('--pred_dirs', type=str, default=pred_dirs, nargs='+')  # path of directory including tracking results

    # target_ids = [6]  # for MOT17-02 pred
    # target_ids = [38]  # for MOT17-02 gt
    # target_ids = [10]  # for MOT17-09 pred
    # target_ids = [7]  # for MOT17-09 gt
    target_ids = [71, 3, 4, 97]  # for MOT20-05 pred
    # target_ids = [836, 470, 490, 881]  # for MOT20-05 gt
    # target_ids = None
    # None: all persons, other numbers: target persons
    parser.add_argument('--target_ids', type=int, default=target_ids, nargs='+')

    # General Arguments
    parser.add_argument('--view-size', type=int, default=None)
    parser.add_argument('--single_img', action='store_true', default=False)
    parser.add_argument('--view_gt', action='store_true', default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_args()
    main(opt)
