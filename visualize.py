import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import keras
import json

from Models.LOUPE import loupe_model

# gpu handling
gpu = '/gpu:' + str(1)
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

models_dir = './trained_models/heart_16_non-task-adaptive/'
dataset_dir = './Data/Task02_Heart_np/'
n_class = 2
image_size = (320, 320)
image_channel = 1
nb_epochs_train = 60
batch_size = 12
num_init_filters = 16
sparsity = 1. / 16
if_train = True
if_task_adaptive = True

model = loupe_model(
    input_shape=image_size + (image_channel,),
    n_class=n_class,
    filt=num_init_filters,
    kern=3,
    sample_slope=200,
    model_type='v2',
    sparsity=sparsity
)

model.load_weights(os.path.join(models_dir, 'my_trained_model.h5'))

with open(os.path.join(dataset_dir, 'dataset.json'), 'r') as f:
    data_dict = json.load(f)
test_list = data_dict['test']

idx = 42
data = np.load(os.path.join(dataset_dir, 'images', '%s.npz' % test_list[idx][0]), mmap_mode='r')
test_image = data['image'][:, :, int(test_list[idx][1]), 0]
test_mask = data['mask'][:, :, int(test_list[idx][1])]
test_image, test_mask = test_image[None, ..., None], test_mask[None, ...]
print(test_image.shape, test_mask.shape)


sub_model = keras.models.Model(model.inputs, model.get_layer('input_layer').output)
img_input = sub_model.predict(test_image)[0].squeeze()
img_input = np.flip(img_input)

sub_model = keras.models.Model(model.inputs, model.get_layer('fft').output)
kspace = sub_model.predict(test_image)[0].squeeze()
kspace = tf.abs(tf.complex(kspace[..., 0], kspace[..., 1]))
kspace = tf.math.log(kspace + 1e-6).numpy()
kspace = np.flip(kspace)

sub_model = keras.models.Model(model.inputs, model.get_layer('sampled_mask').output)
mask_scaled = sub_model.predict(test_image)[0].squeeze()

sub_model = keras.models.Model(model.inputs, model.get_layer('under_sample_kspace').output)
kspace_sampled = sub_model.predict(test_image)[0].squeeze()
kspace_sampled = tf.abs(tf.complex(kspace_sampled[..., 0], kspace_sampled[..., 1]))
kspace_sampled = tf.math.log(kspace_sampled + 1e-6).numpy()
kspace_sampled = np.flip(kspace_sampled)

sub_model = keras.models.Model(model.inputs, model.get_layer('under_sample_img').output)
img_sampled = sub_model.predict(test_image)[0].squeeze()
img_sampled = tf.abs(tf.complex(img_sampled[..., 0], img_sampled[..., 1])).numpy()
img_sampled = np.flip(img_sampled)

sub_model = keras.models.Model(model.inputs, model.get_layer('recon_final').output)
img_recon = sub_model.predict(test_image)[0].squeeze()
img_recon = np.flip(img_recon)

sub_model = keras.models.Model(model.inputs, model.get_layer('seg_unet').output)
seg_pred = sub_model.predict(test_image)[0].squeeze()
seg_pred = tf.math.argmax(seg_pred, axis=-1).numpy()
seg_pred = np.flip(seg_pred)

seg_gt = test_mask[0].squeeze()
seg_gt = np.flip(seg_gt)



plt.figure(figsize=(5.5, 10))
plt.tight_layout()
plt.subplot(4, 2, 1)
plt.imshow(img_input.T, cmap='gray')
plt.title('input')
plt.axis('off')
plt.subplot(4, 2, 2)
plt.imshow(kspace.T, cmap='gray')
plt.title('kspace')
plt.axis('off')
plt.subplot(4, 2, 3)
plt.imshow(mask_scaled.T, cmap='gray')
plt.title('scaled mask')
plt.axis('off')
plt.subplot(4, 2, 4)
plt.imshow(kspace_sampled.T, cmap='gray')
plt.title('sampled kspace')
plt.axis('off')
plt.subplot(4, 2, 5)
plt.imshow(img_sampled.T, cmap='gray')
plt.title('undersampled image')
plt.axis('off')
plt.subplot(4, 2, 6)
plt.imshow(img_recon.T, cmap='gray')
plt.title('reconstructed image')
plt.axis('off')
plt.subplot(4, 2, 7)
plt.imshow(seg_pred.T, cmap='gray')
plt.title('predicted mask')
plt.axis('off')
plt.subplot(4, 2, 8)
plt.imshow(seg_gt.T, cmap='gray')
plt.title('ground-truth mask')
plt.axis('off')
plt.show()


