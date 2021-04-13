import random

import numpy as np
import torch
from PIL import Image
from scipy.interpolate import interp2d
from scipy.ndimage.interpolation import map_coordinates
from skimage.color import heg2rgb, rgb2heg
from skimage.util import img_as_ubyte


# Augmentation dans l'espace HE
class RandomHEaugmentation(object):
    """
    Color augmentation in the H&E color space. Projection of RGB in the hematoxylin eosin 
    and DAB color space

    Args:
        h_std (float): standard deviation of the Gaussian noise added to the hematoxylin 
        channel in the H&E space
        e_std (float): standard deviation of the Gaussian noise added to the eosin 
        channel in the H&E space
    """

    def __init__(self, h_std, e_std):
        super().__init__()
        self.h_std = h_std
        self.e_std = e_std

    def __call__(self, img):
        augmentation = np.array(torch.normal(
            0, torch.Tensor([self.h_std, self.e_std, 0])))
        # Use PyTorch random number generator instead of Numpy's
        # augmentation = np.random.normal(0, [self.h_std, self.e_std, 0], 3)
        img = np.array(img)
        img = rgb2heg(img) + augmentation
        img = img_as_ubyte(heg2rgb(img))
        img = Image.fromarray(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + ' (h_std={}, e_std={})'.format(self.h_std, self.e_std)

###


class RandomRotate90(torch.nn.Module):
    """
    Rotate the given image by an angle of [0, 90, 180, 270]°
    """

    def __init__(self, choice='uniform'):
        super().__init__()
        self.choice = choice

    def _get_param(self):
        if self.choice == 'uniform':
            k = int(torch.randint(0, 4, (1,)))
            k = k*90
            # Use PyTorch random number generator instead of Numpy's
            # k = np.random.choice([0, 90, 180, 270])
        return k

    def forward(self, img):
        angle = self._get_param()
        return img.rotate(angle)

    def __repr__(self):
        return self.__class__.__name__
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
        # Use PyTorch random number generator instead of Numpy's
        # angle = random.randrange(0, 362)
        angle = int(torch.randint(0, 362, (1,)))
        rot_bbox = rotate_rect(bbox, angle)
        if np.min(rot_bbox) > 0 and np.max(rot_bbox) < 512:
            rect_out = False
    return angle


class CropAndRotate(torch.nn.Module):

    def __init__(self, original_size=512, size=224):
        super().__init__()
        self.size = size
        self.original_size = original_size

    def forward(self, x):
        # Use PyTorch random number generator instead of Numpy's
        x1, y1 = random.randrange(0, 288), random.randrange(0, 288)
        x1, y1 = int(torch.randint(0, 288, (1,))), int(
            torch.randint(0, 288, (1,)))
        bbox = np.array([x1, y1, x1 + 224, y1 + 224], dtype=int)
        final_angle = find_angle(bbox)
        x = x.rotate(final_angle)
        x = x.crop(bbox)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (Size={})'.format(self.size)
# Crop


class Crop(torch.nn.Module):
    """
    Crops randomly the given image

    Args:
        original_size (int): Size of the copped image.
        size (int): Desired output size of the crop.
    """

    def __init__(self, original_size=512, size=224,):
        super().__init__()
        self.size = size
        self.original_size = original_size

    def forward(self, img):
        # Use PyTorch random number generator instead of Numpy's
        rand1 = int(torch.randint(0, self.original_size - self.size - 1, (1,)))
        rand2 = int(torch.randint(0, self.original_size - self.size - 1, (1,)))
        img = img.crop((rand1, rand2, rand1 + 224, rand2 + 224))
        return img

    def __repr__(self):
        return self.__class__.__name__ + ' (Size={})'.format(self.size)

# Elastic Distorsion


class MultipleElasticDistort(object):
    """Locally performs elastic distortions on the image.
       It can take a PIL image of any size if it is a square, RGB or grayscale.

       Args:
          cutting : n means the image is divided into a grid of n*n squares, the elastic distortion being applied on each square.
            Beware : cutting has to dvide the size of the input image 

          percentage : value x between 0 and 1, defines the range of stretching / compressing the square, each corner can be moved of
          x*square_size from its original position

      Output : 
        Is a PIL image
    """

    def __init__(self, percentage=0.2, cutting=8):
        self.percentage = percentage
        self.cutting = cutting

    def __call__(self, x):
        size = x.size[0]
        original_coordinates = np.array([[[max(0, i*size//self.cutting-1), max(0, j*size//self.cutting-1)]
                                        for i in range(self.cutting + 1)] for j in range(self.cutting + 1)])
        new_coordinates = original_coordinates.copy()
        r = int(self.percentage*size/self.cutting)
        for i in range(1, self.cutting):
            for j in range(1, self.cutting):  # les points intérieurs
                new_coordinates[i, j][0] += int(torch.randint(-r, r, (1,)))
                new_coordinates[i, j][1] += int(torch.randint(-r, r, (1,)))
        tab = np.asarray(x)
        if len(tab.shape) == 2:
            im = Image.new('L', (size, size))
        else:
            im = Image.new('RGB', (size, size))
        for i in range(self.cutting):
            for j in range(self.cutting):
                i_corners = np.array([0, size//self.cutting])
                j_corners = np.array([0, size//self.cutting])
                crop_i = np.array([new_coordinates[i, j][0], new_coordinates[i+1, j]
                                  [0], new_coordinates[i, j+1][0], new_coordinates[i+1, j+1][0]])
                crop_j = np.array([new_coordinates[i, j][1], new_coordinates[i+1, j]
                                  [1], new_coordinates[i, j+1][1], new_coordinates[i+1, j+1][1]])
                interp_i = interp2d(i_corners, j_corners, crop_i)
                interp_j = interp2d(i_corners, j_corners, crop_j)
                k = np.arange(size//self.cutting)
                l = np.arange(size//self.cutting)
                coords_i = interp_i(k, l)
                coords_j = interp_j(k, l)
                if len(tab.shape) == 2:  # Pour les images en niveau de gris
                    crop_ij = map_coordinates(tab, [coords_j, coords_i])
                else:
                    crop_ij = np.zeros(
                        [coords_j.shape[0], coords_j.shape[1],
                            tab.shape[2]], dtype=np.uint8
                    )
                    for channel in range(tab.shape[2]):
                        crop_ij[:, :, channel] = map_coordinates(
                            tab[:, :, channel], [coords_j, coords_i]
                        )
                im_crop_ij = Image.fromarray(crop_ij)
                im.paste(im_crop_ij, (j*size//self.cutting, i*size//self.cutting))
        return(im)

        def __repr__(self):
            return self.__class__.__name__ + ' (ratio={})'.format(self.size)
