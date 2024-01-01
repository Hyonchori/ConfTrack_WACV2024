# ConfTrack_WACV2024

![conftrack_methods](./assets/conftrack_methods.png)
![conftrack_framework](./assets/conftrack_framework.png)

> #### ConfTrack: Kalman Filter-based Multi-Person Tracking by Utilizing Confidence Score of Detection Box  
> Hyeonchul Jung, Seokjun Kang, Takgen Kim, HyeongKi Kim  
> [Paper.pdf](https://openaccess.thecvf.com/content/WACV2024/papers/Jung_ConfTrack_Kalman_Filter-Based_Multi-Person_Tracking_by_Utilizing_Confidence_Score_of_WACV_2024_paper.pdf)

# Abstract
Kalman filter-based tracking-by-detection (KFTBD) trackers are effective methods for solving multi-person tracking tasks. However, in crowd circumstances, noisy detection results (bounding boxes with low-confidence scores) can cause ID switch and tracking failure of trackers since these trackers utilize the detector’s output directly. In this paper, to solve the problem, we suggest a novel tracker
called ConfTrack based on a KFTBD tracker. Compared with conventional KFTBD trackers, ConfTrack consists of novel algorithms, including low-confidence object penalization and cascading algorithms for effectively dealing with noisy detector outputs. ConfTrack is tested on diverse domains of datasets such as the MOT17, MOT20, DanceTrack, and HiEve datasets. ConfTrack has proved its robustness
in crowd circumstances by achieving the highest score at HOTA and IDF1 metrics in the MOT20 dataset.

# Acknowledgment and Citation
The codebase is built highly upon [BoTSORT](https://github.com/NirAharon/BoT-SORT), [ByteTrack](https://github.com/ifzhang/ByteTrack), [OCSORT](https://github.com/noahcao/OC_SORT), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and [YOLOv5](https://github.com/ultralytics/yolov5). Many thanks for their wonderful works.

    @inproceedings{jung2024conftrack,
    title={ConfTrack: Kalman Filter-Based Multi-Person Tracking by Utilizing Confidence Score of Detection Box},
    author={Jung, Hyeonchul and Kang, Seokjun and Kim, Takgen and Kim, HyeongKi},
    booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
    pages={6583--6592},
    year={2024}
    }
