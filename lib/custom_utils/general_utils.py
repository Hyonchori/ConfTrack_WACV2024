import re
import glob
from pathlib import Path


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def xyxy2xywh(xyxy):
    xywh = xyxy.copy()
    if len(xyxy.shape) == 1:
        xywh[2] = xyxy[2] - xyxy[0]
        xywh[3] = xyxy[3] - xyxy[1]
    else:
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return xywh


def xywh2xyxy(xywh):
    xyxy = xywh.copy()
    if len(xyxy.shape) == 1:
        xyxy[2] = xyxy[0] + xyxy[2]
        xyxy[3] = xyxy[1] + xyxy[3]
    else:
        xyxy[:, 2] = xyxy[:, 0] + xyxy[:, 2]
        xyxy[:, 3] = xyxy[:, 1] + xyxy[:, 3]
    return xyxy
