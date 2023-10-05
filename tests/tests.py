from PIL import ImageDraw, Image
import numpy as np
from ameisedataset import core as ameise
from ameisedataset.utils import transformation as tf
from ameisedataset.data.names import Camera, Lidar
import matplotlib
import matplotlib.pyplot as plt


def show_disparity_map(disparity_map, cmap_name="viridis", val_min=None, val_max=None):
    cmap = matplotlib.colormaps[cmap_name]
    val_min = val_min if val_min is not None else np.min(disparity_map)
    val_max = val_max if val_max is not None else np.max(disparity_map)
    norm_values = (disparity_map - val_min) / (val_max - val_min)
    colored_map = (cmap(norm_values)[:, :, :3] * 255).astype(np.uint8)

    # Konvertieren Sie das NumPy-Array in ein PIL-Bild und zeigen Sie es an
    img = Image.fromarray(colored_map)
    img.show()


def plot_points_on_image(img, points, values, cmap_name="inferno", radius=2, val_min=None, val_max=None):
    draw = ImageDraw.Draw(img)
    cmap = matplotlib.colormaps[cmap_name + "_r"]
    val_min = val_min * 1000 if val_min is not None else np.min(values)
    val_max = val_max * 1000 if val_max is not None else np.max(values)
    norm_values = (values - val_min) / (val_max - val_min)
    for punkt, value in zip(points, norm_values):
        x, y = punkt
        rgba = cmap(value)
        farbe = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=farbe)
    img.show()


infos, frames = ameise.unpack_record("samples/frame.4mse")

pts, proj = tf.get_projection_matrix(frames[-1].lidar[ameise.Lidar.OS1_TOP].points, infos.lidar[ameise.Lidar.OS1_TOP], infos.cameras[Camera.STEREO_LEFT])

image_left = frames[-1].cameras[Camera.STEREO_LEFT]
image_right = frames[-1].cameras[Camera.STEREO_RIGHT]
points_top = frames[-1].lidar[Lidar.OS1_TOP]
print(points_top.get_timestamp())
print(np.amax(points_top.points['t']))

im_rect_l = tf.rectify_image(image_left, infos.cameras[Camera.STEREO_LEFT])
im_rect_r = tf.rectify_image(image_right, infos.cameras[Camera.STEREO_RIGHT])
disparity_map = tf.create_disparity_map(im_rect_l, im_rect_r)

plot_points_on_image(im_rect_l, proj, pts['range'], val_min=8, val_max=50)
show_disparity_map(disparity_map, val_min=0, val_max=100)





