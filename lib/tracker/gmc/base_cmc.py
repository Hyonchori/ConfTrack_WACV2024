import os
import copy
from typing import List

import cv2
import numpy as np

from ..detection.base_detection import BaseDetection


class BaseCMC:
    def __init__(
            self,
            type_cmc: str,  # ['file', 'sparse', 'sift', 'ecc', None]
            downscale: int = 2,
            cmc_result_dir: str = None,
            vid_name: str = None,
            target_split: str = None,  # ['train', 'val', 'test']
    ):
        self.method = type_cmc
        self.downscale = max(1, int(downscale))

        print(f'Loading CMC model "{self.method}" on "{vid_name}"...')
        if self.method == 'orb':
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        elif self.method == 'sift':
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        elif self.method == 'ecc':
            number_of_iteration = 5000
            termination_eps = 1e-6
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iteration, termination_eps)

        elif self.method == 'sparse':
            self.feature_params = dict(maxCorners=3000, qualityLevel=0.01, minDistance=1, blockSize=3,
                                       useHarrisDetector=False, k=0.04)

            # self.gmc_file = open(os.path.join('/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/cmc_files/KITTI_testing',
            #                                   f'GMC-{vid_name}.txt'), 'w')
            # self.gmc_file = open(os.path.join('/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/cmc_files/HiEve_train',
            #                                   f'GMC-{vid_name}.txt'), 'w')
            # self.gmc_file = open(os.path.join('/home/jhc/PycharmProjects/MOT/MOT_study/ConfTrack/cmc_files/HiEve_test',
            #                                   f'GMC-{vid_name}.txt'), 'w')
            # gmc_line = str(0) + "\t" + str(1.000000) + "\t" + str(0.000000) + "\t" + str(
            #     0.000000) + "\t" + str(0.000000) + "\t" + str(1.000000) + "\t" + str(0.000000) + "\n"
            # self.gmc_file.write(gmc_line)

        elif self.method == 'file':
            self.cmc_result_dir = cmc_result_dir
            self.vid_name = vid_name
            self.current_cmc_file = None

            if 'MOT17' in self.vid_name:  # get MOT17 cmc files
                if target_split in ['train', 'test']:
                    self.cmc_result_dir = os.path.join(self.cmc_result_dir, 'MOTChallenge')
                else:  # self.target_split == 'val'
                    self.cmc_result_dir = os.path.join(self.cmc_result_dir, 'MOT17_ablation')

            elif 'MOT20' in self.vid_name:  # get MOT20 cmc files
                if target_split in ['train', 'test']:
                    self.cmc_result_dir = os.path.join(self.cmc_result_dir, 'MOTChallenge')
                else:  # self.target_split == 'val'
                    self.cmc_result_dir = os.path.join(self.cmc_result_dir, 'MOT20_ablation')

            elif 'dancetrack' in self.vid_name:
                self.cmc_result_dir = os.path.join(self.cmc_result_dir, 'DanceTrack')

        elif self.method is None:
            pass

        else:
            raise Exception(f"Given method '{self.method}' is not implemented yet!")
        print(f'\tCMC model loaded!')

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None

        self.initializedFirstFrame = False

    def compute_affine(
            self,
            raw_frame: np.ndarray,  # (height, width, channels)
            detections: List[BaseDetection] = None,
            img_idx: int = None
    ):
        if self.method == 'orb' or self.method == 'sift':
            return self.applyFeatures(raw_frame, detections)
        elif self.method == 'ecc':
            return self.applyEcc(raw_frame)
        elif self.method == 'sparse':
            return self.applySparseOptFlow(raw_frame, img_idx)
        elif self.method == 'file':
            return self.applyFile(img_idx)
        else:
            return np.eye(2, 3)

    def applyEcc(self, raw_frame):
        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        # Downscale image
        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()

            # Initialization done
            self.initializedFirstFrame = True

        try:
            (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
        except:
            print('Warning: find transform failed. Set warp as identity')

        return H

    def applyFeatures(self, raw_frame, detections: List[BaseDetection] = None):
        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image
        if self.downscale > 1.0:
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # find the keypoints
        mask = np.zeros_like(frame)
        mask[int(0.02 * height): int(0.98 * height), int(0.02 * width): int(0.98 * width)] = 255
        if detections is not None:
            for det in detections:
                xyxy = (det.xyxy / self.downscale).astype(np.int_)
                mask[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]] = 0

        keypoints = self.detector.detect(frame, mask)

        # compute the descriptors
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # Match descriptors
        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)

        # Filtered matches based on smallest spatial distance
        matches = []
        spatialDistances = []
        maxSpatialDistance = 0.25 * np.array([width, height])

        # Handle empty matches case
        if len(knnMatches) == 0:
            # Store to next iteration
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)

            return H

        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prevKeyPointLocation = self.prevKeyPoints[m.queryIdx].pt
                currKeyPointLocation = keypoints[m.trainIdx].pt

                spatialDistance = (prevKeyPointLocation[0] - currKeyPointLocation[0],
                                   prevKeyPointLocation[1] - currKeyPointLocation[1])

                if (np.abs(spatialDistance[0]) < maxSpatialDistance[0]) and \
                        (np.abs(spatialDistance[1]) < maxSpatialDistance[1]):
                    spatialDistances.append(spatialDistance)
                    matches.append(m)

        meanSpatialDistances = np.mean(spatialDistances, 0)
        stdSpatialDistances = np.std(spatialDistances, 0)

        inliesrs = (spatialDistances - meanSpatialDistances) < 2.5 * stdSpatialDistances

        goodMatches = []
        prevPoints = []
        currPoints = []
        for i in range(len(matches)):
            if inliesrs[i, 0] and inliesrs[i, 1]:
                goodMatches.append(matches[i])
                prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # Handle downscale
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)

        return H

    def applySparseOptFlow(self, raw_frame: np.ndarray, img_idx: int):
        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # find the keypoints
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        # find correspondences
        matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)

        # leave good correspondences only
        prevPoints = []
        currPoints = []

        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # Handle downscale
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        # gmc_line = str(img_idx) + "\t" + str(H[0, 0]) + "\t" + str(H[0, 1]) + "\t" + str(
        #     H[0, 2]) + "\t" + str(H[1, 0]) + "\t" + str(H[1, 1]) + "\t" + str(H[1, 2]) + "\n"
        # self.gmc_file.write(gmc_line)

        return H

    def applyFile(self, img_idx: int):
        if self.current_cmc_file is None:
            if 'dancetrack' in self.vid_name:
                target_vid = '-'.join([self.vid_name[:-4], self.vid_name[-4:]])
            else:
                target_vid = '-'.join(self.vid_name.split('-')[:2])
            target_path = os.path.join(self.cmc_result_dir, f'GMC-{target_vid}.txt')
            with open(target_path) as f:
                # self.current_cmc_file = {int(x.split('\t')[0]): x[:-2].split('\t')[1:] for x in f.readlines()}
                self.current_cmc_file = {i: x[:-2].split('\t')[1:] for i, x in enumerate(f.readlines())}
        try:
            tmp_cmc = self.current_cmc_file[img_idx]
            H = np.eye(2, 3, dtype=np.float_)
            H[0, 0] = float(tmp_cmc[0])
            H[0, 1] = float(tmp_cmc[1])
            H[0, 2] = float(tmp_cmc[2])
            H[1, 0] = float(tmp_cmc[3])
            H[1, 1] = float(tmp_cmc[4])
            H[1, 2] = float(tmp_cmc[5])
        except KeyError:
            H = np.eye(2, 3, dtype=np.float_)
        return H
