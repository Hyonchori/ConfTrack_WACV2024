# Visualize tracking label of MOT(17, 20) dataset
import argparse
import os
from collections import defaultdict

import cv2
import numpy as np


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


colors = Colors()  # create instance for 'from utils.plots import colors'
CLASSES = {1: "pedestrian", 2: "person on vehicle", 3: "car", 4: "bicycle", 5: "motorbike", 6: "non motorized vehicle",
           7: "static person", 8: "distractor", 9: "occluder", 10: "occluder on the ground", 11: "occluder full",
           12: "reflection", 13: "crowd"}


def parsing_mot_gt(gt_path):
    with open(gt_path) as f:
        rets = [x.strip('\n').split(',') for x in f.readlines()]
        gt = defaultdict(list)
        for ret in rets:
            gt[int(ret[0]) - 1].append(ret[1:])
    return gt


def main(args):
    mot_root = args.mot_root
    target_select = args.target_select
    target_split = args.target_split
    target_vid = args.target_vid
    if target_vid is not None and not isinstance(target_vid, list):
        target_vid = [target_vid]
    target_cls = args.target_cls

    view_size = args.view_size
    view_id = args.view_id
    view_confident = args.view_confident
    visibility_thr = args.visibility_thr
    hide_label = args.hide_label
    hide_visibility = args.hide_visibility

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

    for vid_idx, vid_name in enumerate(vid_list):
        print(f"\n--- Processing {vid_idx + 1} / {len(vid_list)}'s video: {vid_name}")
        vid_dir = os.path.join(vid_root, vid_name)
        img_root = os.path.join(vid_dir, 'img1')
        imgs = sorted(os.listdir(img_root))
        if target_split == 'train' or target_split == 'val':
            gt_path = os.path.join(vid_dir, 'gt', 'gt.txt')
            gt = parsing_mot_gt(gt_path)
        else:
            gt = None

        img_idx = 0
        total_len = len(imgs)

        while img_idx < total_len:
            img_path = os.path.join(img_root, imgs[img_idx])
            img = cv2.imread(img_path)
            plot_info(img, f'{vid_name}: {img_idx + 1} / {len(imgs)}', font_size=2)

            bboxes = gt[img_idx]
            for bbox in bboxes:
                track_id = int(bbox[0])
                trk_color = colors(track_id, True)
                xywh = np.array(list(map(float, bbox[1: 5])))
                xyxy = xywh2xyxy(xywh)
                confidence = float(bbox[5])
                cls_id = int(bbox[6])
                cls_color = colors(cls_id, True)
                cls = CLASSES[cls_id]
                visibility = float(bbox[7])

                if view_id:
                    if confidence == 1.0:
                        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), trk_color, 3)
                        plot_label(img, xyxy, f'{str(track_id)}', trk_color, font_size=0.6)
                else:
                    if view_confident and confidence != 1.0:
                        continue
                    if target_cls and cls_id not in target_cls:
                        continue
                    if visibility >= visibility_thr:
                        if hide_label:
                            label = ''
                        else:
                            label = f'{cls}: {visibility:.2f}' if not hide_visibility else f'{cls}'
                        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), cls_color, 3)
                        plot_label(img, xyxy, label, cls_color, font_size=0.6)

            if view_size is not None:
                img = letterbox(img, view_size, auto=False)[0]

            cv2.imshow(vid_name, img)
            keyboard_input = cv2.waitKey(0) & 0xff
            if keyboard_input == ord('q'):
                break
            elif keyboard_input == 27:  # 27: esc
                import sys
                sys.exit()
            elif keyboard_input == ord('a'):
                img_idx = max(0, img_idx - 1)
            elif keyboard_input == ord('d'):
                img_idx = min(total_len, img_idx + 1)
            else:
                img_idx += 1

        cv2.destroyWindow(vid_name)


def xywh2xyxy(xywh):
    xyxy = xywh.copy()
    xyxy[2] = xyxy[0] + xyxy[2]
    xyxy[3] = xyxy[1] + xyxy[3]
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

    # Arguments for MOT tracking dataset
    mot_root = '/home/jhc/Desktop/dataset/open_dataset/MOT'
    parser.add_argument('--mot_root', type=str, default=mot_root)

    target_select = 'MOT20'
    parser.add_argument('--target_select', type=str, default=target_select)

    target_split = 'val'  # ['train', 'val', 'test']
    parser.add_argument('--target_split', type=str, default=target_split)

    target_vid = 2  # None: all videos, other numbers: target videos
    parser.add_argument('--target_vid', type=int, default=target_vid, nargs='+')

    target_cls = [1, 3, 4]
    parser.add_argument('--target_cls', type=int, default=target_cls, nargs='+')

    # General Arguments
    parser.add_argument("--view-size", type=int, default=[720, 1280])
    parser.add_argument('--view-id', action='store_true', default=True)
    parser.add_argument('--view_confident', action='store', default=False)  # only visualize confident object
    parser.add_argument("--visibility-thr", type=float, default=0.0)
    parser.add_argument("--hide_label", action="store_true", default=True)
    parser.add_argument("--hide-visibility", action="store_true", default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_args()
    main(opt)
