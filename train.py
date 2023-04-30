# imports
import os
import numpy as np
import random
import tensorflow as tf

# loupe
from Models.LOUPE import loupe_model
from DataLoaders.ImagePipeline2D import get_dataset
from Runtime.Run import TrainEngine

###############################################################################
# parameters
###############################################################################

# random seeds
tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

# hyper-parameters
gpu_id = 0  # gpu id
trained_models_dir = './trained_models/heart_32/'  # change this to a location to save models
dataset_dir = './Data/Task02_Heart_np/'
n_class = 2
image_size = (320, 320, 1)
nb_epochs_train = 60
batch_size = 12
sparsity = 1. / 32
if_train = True

if not os.path.exists(trained_models_dir):
    os.mkdir(trained_models_dir)

###############################################################################
# Data
###############################################################################

train_ds, test_ds = get_dataset(
    data_dir=dataset_dir,
    batch_size=batch_size
)

###############################################################################
# GPU
###############################################################################

# gpu handling
gpu = '/gpu:' + str(gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

###############################################################################
# Prepare model
###############################################################################

# model definition
model = loupe_model(
    input_shape=image_size,
    n_class=n_class,
    filt=16,
    kern=3,
    sample_slope=200,
    model_type='v2',
    sparsity=sparsity
)

###############################################################################
# Train model
###############################################################################

print('=' * 20, 'Start Training', '=' * 20)
train_engine = TrainEngine(
    model=model,
    train_ds=train_ds,
    val_ds=test_ds,
    n_epochs=nb_epochs_train,
    n_class=n_class,
    loss_weights=[0.5, 0.5],
    val_freq=10
)
if if_train:
    train_engine.train()
    model.save_weights(os.path.join(trained_models_dir, 'my_trained_model.h5'))

print('=' * 20, 'Test', '=' * 20)
model.load_weights(os.path.join(trained_models_dir, 'my_trained_model.h5'))
train_engine.evaluate()



