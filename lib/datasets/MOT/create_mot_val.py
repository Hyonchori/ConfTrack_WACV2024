import argparse
import os
import shutil
import time

from tqdm import tqdm
from pycocotools.coco import COCO


def main(args):
    json_path = args.json_path
    mot17_root = args.mot17_root
    trackeval_gt_root = args.trackeval_gt_root
    save = args.save

    save_dir = os.path.join(mot17_root, 'val')
    trackeval_save_dir = os.path.join(trackeval_gt_root, 'MOT17-val')
    if save:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if not os.path.isdir(trackeval_save_dir):
            os.mkdir(trackeval_save_dir)

    coco = COCO(json_path)

    # copy val images for prediction
    print('\nMaking val dataset...')
    img_ids = coco.getImgIds()
    img_infos = coco.loadImgs(img_ids)

    for i, (img_id, img_info) in tqdm(enumerate(zip(img_ids, img_infos)), total=len(img_infos)):
        vid_name, dir_name, img_name = img_info['file_name'].split('/')
        target_img_path = os.path.join(mot17_root, 'train', vid_name, dir_name, img_name)
        if save:
            target_save_dir = os.path.join(save_dir, vid_name, dir_name)
            if not os.path.isdir(target_save_dir):
                os.makedirs(target_save_dir)
            target_save_path = os.path.join(target_save_dir, img_name)
            shutil.copyfile(target_img_path, target_save_path)

    # copy val gt for evaluation
    time.sleep(0.5)
    print('\nMaking val GT...')
    vid_names = os.listdir(save_dir)

    for vid_name in tqdm(vid_names):
        tmp_img_dir = os.path.join(save_dir, vid_name, 'img1')
        img_indices = [int(x.split('.')[0]) for x in os.listdir(tmp_img_dir)]
        tmp_start_idx = min(img_indices)
        tmp_end_idx = max(img_indices)

        tmp_det_path = os.path.join(mot17_root, 'train', vid_name, 'det', 'det.txt')
        tmp_gt_path = os.path.join(mot17_root, 'train', vid_name, 'gt', 'gt.txt')
        tmp_seqinfo_path = os.path.join(mot17_root, 'train', vid_name, 'seqinfo.ini')

        save_det_dir = os.path.join(save_dir, vid_name, 'det')
        save_gt_dir = os.path.join(save_dir, vid_name, 'gt')
        save_gt_trackeval_dir = os.path.join(trackeval_save_dir, vid_name, 'gt')
        if save:
            if not os.path.isdir(save_det_dir):
                os.makedirs(save_det_dir)
            if not os.path.isdir(save_gt_dir):
                os.makedirs(save_gt_dir)
            if not os.path.isdir(save_gt_trackeval_dir):
                os.makedirs(save_gt_trackeval_dir)

            new_gt = ''
            with open(tmp_gt_path) as f:
                gt = [x for x in f.readlines()]
                for line in gt:
                    if int(line.split(',')[0]) >= tmp_start_idx:
                        new_line = line.split(',')
                        new_line[0] = str(int(new_line[0]) - tmp_start_idx + 1)
                        new_gt += ','.join(new_line)
            save_gt_path = os.path.join(save_gt_dir, 'gt.txt')
            save_gt_trackeval_path = os.path.join(save_gt_trackeval_dir, 'gt.txt')
            with open(save_gt_path, 'w') as f:
                f.write(new_gt)
            with open(save_gt_trackeval_path, 'w') as f:
                f.write(new_gt)

            new_det = ''
            with open(tmp_det_path) as f:
                det = [x for x in f.readlines()]
                for line in det:
                    if int(line.split(',')[0]) >= tmp_start_idx:
                        new_line = line.split(',')
                        new_line[0] = str(int(new_line[0]) - tmp_start_idx + 1)
                        new_det += ','.join(new_line)
            save_det_path = os.path.join(save_det_dir, 'det.txt')
            with open(save_det_path, 'w') as f:
                f.write(new_det)

            new_seqinfo = ''
            with open(tmp_seqinfo_path) as f:
                seqinfo = [x for x in f.readlines()]
                for line in seqinfo:
                    if 'seqLength' not in line:
                        new_seqinfo += line
                    else:
                        new_seqinfo += f'seqLength={tmp_end_idx - tmp_start_idx + 1}\n'
            save_seqinfo_path = os.path.join(save_dir, vid_name, 'seqinfo.ini')
            save_seqinfo_trackeval_path = os.path.join(trackeval_save_dir, vid_name, 'seqinfo.ini')
            with open(save_seqinfo_path, 'w') as f:
                f.write(new_seqinfo)
            with open(save_seqinfo_trackeval_path, 'w') as f:
                f.write(new_seqinfo)


def get_args():
    parser = argparse.ArgumentParser()

    json_path = '/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack_WACV2024/lib/datasets/MOT/mot20_val_half.json'
    parser.add_argument('--json_path', type=str, default=json_path)

    mot17_root = '/home/jhc/Desktop/dataset/open_dataset/mot_test/MOT20'
    parser.add_argument('--mot17_root', type=str, default=mot17_root)

    trackeval_gt_root = '/home/jhc/Desktop/dataset/open_dataset/mot_test/data/gt/mot_challenge'
    parser.add_argument('--trackeval_gt_root', type=str, default=trackeval_gt_root)

    save = True
    parser.add_argument('--save', action='store_true', default=save)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()
    main(opt)
