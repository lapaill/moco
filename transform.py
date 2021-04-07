import random

import numpy as np
from PIL import Image
from scipy.interpolate import interp2d
from scipy.ndimage.interpolation import map_coordinates
from skimage.color import heg2rgb, rgb2heg
from skimage.util import img_as_ubyte


# Augmentation dans l'espace HE
class RandomHEaugmentation(object):
    def __init__(self, h_std, e_std):
        self.h_std = h_std
        self.e_std = e_std

    def __call__(self, img):
        augmentation = np.random.normal(0, [self.h_std, self.e_std, 0], 3)
        img = np.array(img)
        img = rgb2heg(img) + augmentation
        img = img_as_ubyte(heg2rgb(img))
        img = Image.fromarray(img)
        return img


class RandomRotate90(object):
    def __init__(self, choice='uniform'):
        self.choice = choice

    def _get_param(self):
        if self.choice == 'uniform':
            k = np.random.choice([0, 90, 180, 270])
        return k

    def __call__(self, img):
        angle = self._get_param()
        return img.rotate(angle)

### Crop and Rotate


def rotate_rect(bbox, angle):
    """
    Returns coordinates of a bounding box after rotation

    Parameters
    ----------

    bbox : list
        input bounding box
    angle : int 
        rotation angle in degrees

    Returns
    -------
    np.array
        coordinates of the rotated bbox[
        [x1, x1, x2, x2],
        [y1, y2, y2, y1]]
      """
    rad_angle = angle*np.pi/180
    bbox_centered = np.array(bbox - 256)
    x1, y1, x2, y2 = (
        bbox_centered[0], bbox_centered[1], bbox_centered[2], bbox_centered[3])
    rect_bbox = np.array([
        [x1, x1, x2, x2],
        [y1, y2, y2, y1]])

    rot = np.array([
        [np.cos(rad_angle), -np.sin(rad_angle)],
        [np.sin(rad_angle), np.cos(rad_angle)]])  # define the rotation matrix

    rotated_rect = np.dot(rot, rect_bbox) + 256
    return np.array(rotated_rect, dtype=int)


def find_angle(bbox):
    """
    Returns a random angle that ensures that the rotated bounding box lays 
    within the frame

    Parameters
    ----------

    bbox : list
        input bounding box

    Returns
    -------
    int
      """
    rect_out = True
    while rect_out:
        angle = random.randrange(0, 362)
        rot_bbox = rotate_rect(bbox, angle)
        if np.min(rot_bbox) > 0 and np.max(rot_bbox) < 512:
            rect_out = False
    return angle


class CropAndRotate(object):

    def __init__(self, original_size=512, size=224):
        self.size = size
        self.original_size = original_size

    def __call__(self, x):
        x1, y1 = random.randrange(0, 288), random.randrange(0, 288)
        bbox = np.array([x1, y1, x1 + 224, y1 + 224], dtype=int)
        final_angle = find_angle(bbox)
        x = x.rotate(final_angle)
        x = x.crop(bbox)
        return x

# Crop


class Crop(object):

    def __init__(self, original_size=512, size=224):
        self.size = size
        self.original_size = original_size

    def __call__(self, x):
        rand1 = random.randint(0, self.original_size - self.size - 1)
        rand2 = random.randint(0, self.original_size - self.size - 1)
        x = x.crop((rand1, rand2, rand1 + 224, rand2 + 224))
        return x

# Elastic Distorsion


class ElasticDistortion(object):  # attention : pas compatible avec rotation
    """Get distorted crop from image.

    Note that crop corner coordinates should be given following
    mathematics conventions, and not image processing conventions.

    Arguments:
        im: input image (2D, with 1 or more channels).
        crop_x, crop_y: coordinates of four points.
        shape: crop shape.
    Returns:
        crop: image containing resulting crop.
    """

    def __init__(self, original_size=512, size=224, r=50):  # r < crop_size / 2
        self.size = size
        self.original_size = original_size
        self.r = r

    def __call__(self, x):
        i_corners = np.array([0, self.size])
        j_corners = np.array([0, self.size])
        rand_stretch1 = random.randint(-self.r, self.r)
        rand_stretch2 = random.randint(-self.r, self.r)
        rand_stretch3 = random.randint(-self.r, self.r)
        rand_stretch4 = random.randint(-self.r, self.r)
        rand_stretch5 = random.randint(-self.r, self.r)
        rand_stretch6 = random.randint(-self.r, self.r)
        L = [rand_stretch1, rand_stretch2, rand_stretch3,
             rand_stretch4, rand_stretch5, rand_stretch6]
        maxi = max(L)
        # attention au stretch on peut pas partir de 0 ... dépend de la valeur générée par stretch. Idem on peut pas aller jusqu'au bout...
        rand1 = random.randint(maxi, self.original_size - self.size - 1 - maxi)
        rand2 = random.randint(maxi, self.original_size - self.size - 1 - maxi)
        # on ne peut pas dépasser im_size = 512. Il faut générer toutes les coordonnées en random + stretch
        crop_i = np.array([rand1, rand1 + rand_stretch2, rand1 +
                          224 + rand_stretch3, rand1 + 224 + rand_stretch5])
        crop_j = np.array([rand2, rand2 + 224 + rand_stretch1,
                          rand2 + rand_stretch4, rand2 + 224 + rand_stretch6])
        interp_i = interp2d(i_corners, j_corners, crop_i)
        interp_j = interp2d(i_corners, j_corners, crop_j)

        i = np.arange(self.size)
        j = np.arange(self.size)
        coords_i = interp_i(i, j)
        coords_j = interp_j(i, j)
        tab = np.asarray(x)
        if len(tab.shape) == 2:  # Pour les images en niveau de gris
            crop = map_coordinates(tab, [coords_j, coords_i])
        else:
            crop = np.zeros(
                [coords_j.shape[0], coords_j.shape[1], tab.shape[2]], dtype=np.uint8
            )
            for channel in range(tab.shape[2]):
                crop[:, :, channel] = map_coordinates(
                    tab[:, :, channel], [coords_j, coords_i]
                )
        crop = Image.fromarray(crop)
        return crop
