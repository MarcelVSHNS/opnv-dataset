import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, List
from ameisedataset.data import CameraInformation, LidarInformation

def extract_translation_and_euler_from_matrix(mtx):
    # Extrahieren des Translationsvektors
    translation_vector = mtx[:3, 3]

    # Extrahieren der Rotationsmatrix und Umwandeln in Euler-Winkel (Radiant)
    rotation_matrix = mtx[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    euler_angles_rad = rotation.as_euler('xyz', degrees=False)

    return translation_vector, euler_angles_rad


def get_points_on_image(pcloud: List[np.ndarray], lidar_info: LidarInformation, cam_info: CameraInformation,
                        get_valid_only=True,
                        dtype_points_return=None) -> Tuple[np.array, List[Tuple]]:
    """Retrieve the projection matrix based on provided parameters."""
    if dtype_points_return is None:
        dtype_points_return = ['x', 'y', 'z', 'intensity', 'range']
    lidar_tf = Transformation('tbd', 'tbd', lidar_info.extrinsic.xyz, lidar_info.extrinsic.rpy)
    camera_tf = Transformation('tbd', 'tbd', cam_info.extrinsic.xyz, cam_info.extrinsic.rpy)
    camera_inverse_tf = camera_tf.invert_transformation()
    lidar_to_cam_tf = camera_inverse_tf.add_transformation(lidar_tf)
    rect_mtx = np.eye(4)
    rect_mtx[:3, :3] = cam_info.rectification_mtx
    proj_mtx = cam_info.projection_mtx

    projection = []
    points = []
    for point in pcloud:
        point_vals = np.array(point.tolist()[:3])
        # Transform points to new coordinate system
        point_in_camera = proj_mtx.dot(rect_mtx.dot(lidar_to_cam_tf.transformation_mtx.dot(np.append(point_vals[:3], 1))))
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


class Transformation:
    def __init__(self, at, to, xyz, rpy):
        self._at = at
        self._to = to
        self._translation = xyz
        self._rotation = rpy
        self._update_transformation_matrix()

    @property
    def at(self):
        return self._at

    @at.setter
    def at(self, value):
        self._at = value

    @property
    def to(self):
        return self._to

    @to.setter
    def to(self, value):
        self._to = value

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, value):
        self._translation = np.array(value, dtype=float)
        self._update_transformation_matrix()

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = np.array(value, dtype=float)
        self._update_transformation_matrix()

    def _update_transformation_matrix(self):
        rotation = R.from_euler('xyz', self._rotation, degrees=False)
        rotation_matrix = rotation.as_matrix()
        self.transformation_mtx = np.identity(4)
        self.transformation_mtx[:3, :3] = rotation_matrix
        self.transformation_mtx[:3, 3] = self._translation

    def add_transformation(self, transformation_to_add):
        transformation_mtx_to_add = transformation_to_add.transformation_mtx
        new_transformation_mtx = np.dot(self.transformation_mtx, transformation_mtx_to_add)

        translation_vector, euler_angles = extract_translation_and_euler_from_matrix(new_transformation_mtx)

        new_transformation = Transformation(self.at, transformation_to_add.to, translation_vector, euler_angles)

        return new_transformation

    def invert_transformation(self):
        inverse_rotation_matrix = self.transformation_mtx[:3, :3].T
        inverse_translation_vector = -inverse_rotation_matrix @ self.transformation_mtx[:3, 3]
        inverse_transformation_matrix = np.identity(4)
        inverse_transformation_matrix[:3, :3] = inverse_rotation_matrix
        inverse_transformation_matrix[:3, 3] = inverse_translation_vector

        translation_vector, euler_angles = extract_translation_and_euler_from_matrix(inverse_transformation_matrix)

        inverse_transformation = Transformation(self.to, self.at, translation_vector, euler_angles)

        return inverse_transformation

    def __repr__(self):
        translation_str = ', '.join(f"{coord:.3f}" for coord in self.translation)
        rotation_str = ', '.join(f"{angle:.3f}" for angle in self.rotation)
        return (f"Transformation at {self._at} to {self._to},\n"
                f"  translation=[{translation_str}],\n"
                f"  rotation=[{rotation_str}]\n")