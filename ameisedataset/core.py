import dill
import os
from typing import List, Tuple

import ameisedataset.utils.transformation
from ameisedataset.metadata import CameraInformation, LidarInformation, Camera, Lidar, Frame, Infos
from ameisedataset.miscellaneous import compute_checksum, InvalidFileTypeError, ChecksumError, SHA256_CHECKSUM_LENGTH, INT_LENGTH


# Reading modules
def _read_header(file) -> list:
    """Read and deserialize the header from the file."""
    header_length = int.from_bytes(file.read(INT_LENGTH), 'big')
    header_bytes = file.read(header_length)
    return dill.loads(header_bytes)


def _read_info_object(file, name, meta_infos):
    """Read and deserialize an info object from the file."""
    info_length = int.from_bytes(file.read(INT_LENGTH), 'big')
    info_checksum = file.read(SHA256_CHECKSUM_LENGTH)  # SHA-256 checksum length
    info_bytes = file.read(info_length)
    # Verify checksum
    if compute_checksum(info_bytes) != info_checksum:
        raise ChecksumError(f"Checksum of {name} is not correct! Check file.")
    # Deserialize the Info object
    if Camera.is_type_of(name.upper()):
        meta_infos.camera[Camera[name.upper()]] = CameraInformation.from_bytes(info_bytes)
    elif Lidar.is_type_of(name.upper()):
        meta_infos.lidar[Lidar[name.upper()]] = LidarInformation.from_bytes(info_bytes)


def _read_frame_object(file, meta_infos):
    """Read and deserialize a frame from the file."""
    compressed_data_len = int.from_bytes(file.read(INT_LENGTH), 'big')
    compressed_data_checksum = file.read(SHA256_CHECKSUM_LENGTH)  # SHA-256 checksum length
    compressed_data = file.read(compressed_data_len)
    # Verify checksum
    if compute_checksum(compressed_data) != compressed_data_checksum:
        raise ChecksumError("Checksum mismatch. Data might be corrupted!")
    return Frame.from_bytes(compressed_data, pts_dtype=meta_infos.lidar[Lidar.OS1_TOP].dtype)


def unpack_record(filename) -> Tuple[Infos, List[Frame]]:
    """Unpack an AMEISE record file and extract meta information and frames.
        Args:
            filename (str): Path to the AMEISE record file.
        Returns:
            Tuple[Infos, List[Frame]]: Meta information and a list of frames.
        """
    # Ensure the provided file has the correct extension
    if os.path.splitext(filename)[1] != ".4mse":
        raise InvalidFileTypeError("This is not a valid AMEISE-Record file.")
    frames: List[Frame] = []
    meta_infos = Infos(filename)
    with open(filename, 'rb') as file:
        info_names = _read_header(file)
        # Read infos
        for name in info_names:
            _read_info_object(file, name, meta_infos)
        # Read num frames
        num_frames = int.from_bytes(file.read(4), 'big')
        # Read frames
        for _ in range(num_frames):
            frames.append(_read_frame_object(file, meta_infos))
    return meta_infos, frames


# proj_top_left = ameisedataset.get_projection_matrix(Lidar.OS1_TOP, Camera.STEREO_LEFT)              # get projection mtx (4000, 3) idx, x, y
# proj_point_list = ameisedataset.merge_projection_with_pcloud(Frame.lidar.points, proj_top_left)     # delete all points but image an add proj
# 4000, 11
