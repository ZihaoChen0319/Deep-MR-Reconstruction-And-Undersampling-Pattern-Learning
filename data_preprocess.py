import os
import numpy as np
import SimpleITK as sitk
import json
import random
import matplotlib.pyplot as plt

from Preprocessing.Resample import sitk_resample
from Preprocessing.PadCrop import pad_crop


# dir, name and division
random.seed(42)
data_dir = './Data'
data_name = 'Task02_Heart'
val_ratio = 0.2

# mkdir
if not os.path.exists(os.path.join(data_dir, '%s_np' % data_name)):
    os.mkdir(os.path.join(data_dir, '%s_np' % data_name))
if not os.path.exists(os.path.join(data_dir, '%s_np' % data_name, 'images')):
    os.mkdir(os.path.join(data_dir, '%s_np' % data_name, 'images'))

# load dataset information
print('=' * 20, 'Dataset Information', '=' * 20)
if os.path.exists(os.path.join(data_dir, data_name, 'dataset.json')):
    with open(os.path.join(data_dir, data_name, 'dataset.json'), 'r') as f:
        data_dict = json.load(f)
    if 'numTraining' in data_dict.keys(): del data_dict['numTraining']
    if 'numTest' in data_dict.keys(): del data_dict['numTest']
    if 'training' in data_dict.keys(): del data_dict['training']
    if 'test' in data_dict.keys(): del data_dict['test']
else:
    data_dict = {}
print(data_dict)

# divide data to training set and validation set
print('=' * 20, 'Data Division', '=' * 20)
files_list = os.listdir(os.path.join(data_dir, data_name, 'imagesTr'))
files_list = [x.split('.')[0] for x in files_list if x[0] != '.']
random.shuffle(files_list)
test_files_list = files_list[:int(val_ratio * len(files_list))]
train_files_list = files_list[int(val_ratio * len(files_list)):]
print('%d for training, %d for test' % (len(train_files_list), len(test_files_list)))


# preprocess: interpolation, intensity normalization, pad and crop, save to npz, save json file
print('=' * 20, 'Data Processing', '=' * 20)

# collect the statistics about spacing and size, then compute target spacing and size
spacing_list = []
size_list = []
for file_name in files_list:
    image_sitk = sitk.ReadImage(os.path.join(data_dir, data_name, 'imagesTr', '%s.nii.gz' % file_name))
    spacing_list.append(np.array(image_sitk.GetSpacing()))
    size_list.append(np.array(image_sitk.GetSize()))
target_spacing = np.median(np.array(spacing_list), axis=0)[:3]
target_size = np.median(np.array(size_list), axis=0)[:2]
target_size = np.ceil(target_size / 16).astype(int) * 16  # make the size a multiple of 16
data_dict['image_size'] = [int(target_size[i]) for i in range(len(target_size))]
data_dict['image_spacing'] = [target_spacing[i].astype(float) for i in range(len(target_spacing))]
print('Target size', target_size)
print('Target spacing', target_spacing)

train_list, test_list = [], []
for file_name in files_list:
    image_sitk = sitk.ReadImage(os.path.join(data_dir, data_name, 'imagesTr', '%s.nii.gz' % file_name))
    mask_sitk = sitk.ReadImage(os.path.join(data_dir, data_name, 'labelsTr', '%s.nii.gz' % file_name))

    # resampling
    image_sitk = sitk_resample(image_sitk, target_spacing=target_spacing, is_label=False)
    mask_sitk = sitk_resample(mask_sitk, target_spacing=target_spacing, is_label=True)

    image = sitk.GetArrayFromImage(image_sitk)
    if len(image.shape) == 3:
        image = image[None, ...]
    mask = sitk.GetArrayFromImage(mask_sitk)
    image, mask = image.transpose((3, 2, 1, 0)), mask.transpose((2, 1, 0))

    # clipping and z-score normalization of foreground
    nz_indices = np.nonzero(np.abs(image) > 0.1)
    foreground_mask = np.zeros_like(image, dtype=np.uint8)
    foreground_mask[nz_indices] = 1
    for mod_idx in range(image.shape[-1]):
        nz = np.nonzero(foreground_mask[..., mod_idx])
        values = image[..., mod_idx][nz]
        percentile_min, percentile_max = np.percentile(values, [0.5, 99.5])
        values = np.clip(values, a_min=percentile_min, a_max=percentile_max)
        image[..., mod_idx][nz] = (values - np.min(values)) / (np.max(values) - np.min(values))

    # pad or crop
    image = pad_crop(image, target_size=target_size)
    mask = pad_crop(mask, target_size=target_size)

    # collect slices that mask is not all 0
    slice_idx = np.nonzero(np.sum(mask, axis=(0, 1)))[0]
    if file_name in train_files_list:
        for i in slice_idx:
            train_list.append((file_name, str(i)))
    elif file_name in test_files_list:
        for i in slice_idx:
            test_list.append((file_name, str(i)))
    else:
        raise ValueError('%s' % file_name)

    # save as npz
    image = image.astype(np.float32)
    mask = mask.astype(np.uint8)
    foreground_mask = foreground_mask.astype(np.uint8)
    np.savez(os.path.join(data_dir, '%s_np' % data_name, 'images', '%s.npz' % file_name),
             image=image, mask=mask, foreground_mask=foreground_mask)

data_dict['train'] = train_list
data_dict['test'] = test_list
data_dict['numTrain'] = len(train_list)
data_dict['numTest'] = len(test_list)

# save the dataset information json file
with open(os.path.join(data_dir, '%s_np' % data_name, 'dataset.json'), 'w', encoding='utf-8') as f:
    json.dump(data_dict, f, ensure_ascii=False, indent=4)



