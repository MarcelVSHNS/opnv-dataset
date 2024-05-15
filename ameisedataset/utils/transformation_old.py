import numpy as np
import cv2
from typing import Tuple, List

from ameisedataset.data import Pose, CameraInformation, LidarInformation, Image

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles to rotation matrix."""
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    R_x = np.array([[1, 0, 0],
                    [0, cos_r, -sin_r],
                    [0, sin_r, cos_r]])

    R_y = np.array([[cos_p, 0, sin_p],
                    [0, 1, 0],
                    [-sin_p, 0, cos_p]])

    R_z = np.array([[cos_y, -sin_y, 0],
                    [sin_y, cos_y, 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def create_transformation_matrix(translation, rotation):
    """Create a 4x4 homogeneous transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = euler_to_rotation_matrix(rotation[0], rotation[1], rotation[2])
    T[:3, 3] = translation
    return T


def get_points_on_image(pcloud: List[np.ndarray], lidar_info: LidarInformation, cam_info: CameraInformation, get_valid_only=True,
                        dtype_points_return=None) -> Tuple[np.array, List[Tuple]]:
    """Retrieve the projection matrix based on provided parameters."""
    if dtype_points_return is None:
        dtype_points_return = ['x', 'y', 'z', 'intensity', 'range']
    lidar_to_cam_tf_mtx = transform_to_sensor(lidar_info.extrinsic, cam_info.extrinsic)
    rect_mtx = np.eye(4)
    rect_mtx[:3, :3] = cam_info.rectification_mtx
    proj_mtx = cam_info.projection_mtx

    projection = []
    points = []
    for point in pcloud:
        point_vals = np.array(point.tolist()[:3])
        # Transform points to new coordinate system
        point_in_camera = proj_mtx.dot(rect_mtx.dot(lidar_to_cam_tf_mtx.dot(np.append(point_vals[:3], 1))))
        # check if pts are behind the camera
        u = point_in_camera[0] / point_in_camera[2]
        v = point_in_camera[1] / point_in_camera[2]
        if get_valid_only:
            if point_in_camera[2] <= 0:
                continue
            elif 0 <= u < cam_info.shape[0] and 0 <= v < cam_info.shape[1]:
                projection.append((u, v))
                points.append(point[dtype_points_return])
            else:
                continue
        else:
            projection.append((u, v))
    return np.array(points, dtype=points[0].dtype), projection


def transform_to_sensor(sensor1: Pose, sensor2: Pose):
    """Transform the data to the sensor's coordinate frame."""
    # Creating transformation matrices
    t1 = create_transformation_matrix(sensor1.xyz, sensor1.rpy)
    t2 = create_transformation_matrix(sensor2.xyz, sensor2.rpy)

    # Computing the transformation from the second sensor (new origin) to the first sensor
    t2_to_1 = np.dot(np.linalg.inv(t2), t1)
    return t2_to_1