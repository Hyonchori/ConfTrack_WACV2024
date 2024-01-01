import numpy as np


class BaseKalmanFilter:
    def __init__(
            self,  # m: state_dim, n: measurement_dim
            system_matrix: np.ndarray,  # shape: (m, m)
            projection_matrix: np.ndarray,  # shape: (n, m)
            init_measure: np.ndarray,  # shape: (n, 1)
            init_confidence: float,
            init_cls: int,
            std_weight_position: float = 1. / 20,
            std_weight_velocity: float = 1. / 160,
            use_NSAK: bool = False,
            nsa_amplify_factor: float = None
    ):
        # matrices for Kalman prediction and update
        self.A = system_matrix
        self.H = projection_matrix

        # init measurement and options
        self.z = init_measure
        self.conf = init_confidence
        self.cls = init_cls
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity
        self.use_NSAK = use_NSAK
        self.nsa_amplify_factor = nsa_amplify_factor

        self.x = None
        self.x_cov = None

    def initialize_state(self, width, height):
        # initialize state and state error covariance
        x = np.zeros([self.A.shape[0], 1], dtype=np.float32)  # Kalman state: (m, 1)
        x[: self.H.shape[0]] = self.z

        std = [
            2 * self._std_weight_position * width,
            2 * self._std_weight_position * height,
            2 * self._std_weight_position * width,
            2 * self._std_weight_position * height,
            10 * self._std_weight_velocity * width,
            10 * self._std_weight_velocity * height,
            10 * self._std_weight_velocity * width,
            10 * self._std_weight_velocity * height
        ]
        x_cov = np.diag(np.square(std, dtype=np.float))  # Kalman state error covariance: (m, m)
        self.x = x
        self.x_cov = x_cov
        return self.x.copy(), self.x_cov.copy()

    def predict(self, width, height, use_CPLT: bool = False):
        # Kalman prediction: predict state(t) using state(t-1)
        if use_CPLT:
            ''' Constant Prediction on Lost Track '''
            self.x[6:] = 0.0  # make state [cx, cy, width, height, cx', cy', 0, 0]

        self.x = np.matmul(self.A, self.x)

        std = [
            self._std_weight_position * width,
            self._std_weight_position * height,
            self._std_weight_position * width,
            self._std_weight_position * height,
            self._std_weight_velocity * width,
            self._std_weight_velocity * height,
            self._std_weight_velocity * width,
            self._std_weight_velocity * height
        ]
        Q = np.diag(np.square(std, dtype=np.float))  # system noise matrix (m, m)
        self.x_cov = np.linalg.multi_dot([self.A, self.x_cov, self.A.T]) + Q
        return self.x.copy(), self.x_cov.copy()

    def measure(self, new_z, new_conf):
        # update measurement
        self.z = new_z
        self.conf = new_conf

    def project(self, width, height):
        # Kalman projection: project state-space to measurement-space
        projected_x = np.matmul(self.H, self.x)

        std = [
            self._std_weight_position * width,
            self._std_weight_position * height,
            self._std_weight_position * width,
            self._std_weight_position * height,
        ]
        R = np.diag(np.square(std, dtype=np.float))  # measurement noise matrix (n, n)

        ''' Noise Scale Adaptive Kalman Filter  '''
        if self.use_NSAK:
            ''' When track on MOT(only pedestrian) '''
            if self.cls == 0:
                R *= (1. - self.conf) * self.nsa_amplify_factor

            ''' When track on KITTI '''
            # if KITTI_CLASSES[self.cls] == 'Person':
            #     R *= (1. - self.conf) * self.nsa_amplify_factor
            # else:
            #     R *= (1. - self.conf) * 1.0

        projected_x_cov = np.linalg.multi_dot([self.H, self.x_cov, self.H.T]) + R
        return projected_x, projected_x_cov

    def update(self, width, height):
        # Kalman update: calculate present state(t) using prediction(t) and prior state(t-1)
        projected_x, projected_x_cov = self.project(width, height)
        K = np.linalg.multi_dot([self.x_cov, self.H.T, np.linalg.inv(projected_x_cov)])  # Kalman gain
        y = self.z - projected_x
        self.x = self.x + np.matmul(K, y)
        self.x_cov = np.matmul(np.identity(self.A.shape[0], dtype=np.float32) - np.matmul(K, self.H), self.x_cov)
        return self.x.copy(), self.x_cov.copy()


KITTI_CLASSES = {1: 'Car', 2: 'Car', 3: 'Car',
                 4: 'Person', 5: 'Person', 6: 'Person', 7: 'Cyclist',
                 8: 'Tram', 9: 'Misc', 10: 'DontCare'}
