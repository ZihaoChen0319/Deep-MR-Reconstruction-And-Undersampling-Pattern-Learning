import numpy as np


def pad_crop(image, target_size):
    for i in range(len(target_size)):
        image = pad_crop_along_axis(image, target_size=target_size[i], axis=i)
    return image


def pad_crop_along_axis(image, target_size, axis):
    size = image.shape[axis]
    if size < target_size:
        diff = target_size - size
        pad_width = [np.floor(diff / 2).astype(int), np.ceil(diff / 2).astype(int)]
        pad_width = axis * [[0, 0]] + [pad_width] + [[0, 0]] * (len(image.shape) - axis - 1)
        image = np.pad(image, pad_width=pad_width, mode='constant', constant_values=0)
    elif size > target_size:
        diff = size - target_size
        start = diff // 2
        image = image.take(indices=np.arange(start, start + target_size).astype(int), axis=axis)
    else:
        pass

    return image


