import numpy as np
import tensorflow as tf
import json
import os


def get_samples_list(data_dir):
    with open(os.path.join(data_dir, 'dataset.json'), 'r') as f:
        data_dict = json.load(f)
    print('=' * 20, 'Dataset Information', '=' * 20)
    for key, value in data_dict.items():
        if key not in ['train', 'test']:
            print('%s:' % key, value)
    return data_dict['train'], data_dict['test']


class ParseFunction:
    def __init__(self, data_dir, get_foreground=False):
        self.data_dir = data_dir
        self.get_foreground = get_foreground

    def parse(self, x):
        data = np.load(os.path.join(self.data_dir, 'images', '%s.npz' % x[0].decode()), mmap_mode='r')
        image = data['image'][:, :, int(x[1].decode()), 0]
        mask = data['mask'][:, :, int(x[1].decode())]
        if self.get_foreground:
            foreground_mask = data['foreground_mask'][:, :, int(x[1].decode()), 0]
            return image[..., None], mask, foreground_mask[..., None]
        else:
            return image[..., None], mask


def configure_for_performance(dataset, batch_size, buffer_size, shuffle=False):
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=buffer_size)
    return dataset


def get_dataset(data_dir, batch_size=16, buffer_size=1000, num_parallel_calls=4, get_foreground=False):
    train_list, test_list = get_samples_list(data_dir)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_list)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_list)
    parse_func = ParseFunction(data_dir=data_dir, get_foreground=get_foreground)
    train_dataset = train_dataset.map(
        lambda x: tf.numpy_function(parse_func.parse, [x], [np.float, np.uint8]),
        num_parallel_calls=num_parallel_calls)
    test_dataset = test_dataset.map(
        lambda x: tf.numpy_function(parse_func.parse, [x], [np.float, np.uint8]),
        num_parallel_calls=num_parallel_calls)
    train_dataset = configure_for_performance(
        train_dataset, batch_size=batch_size, buffer_size=buffer_size, shuffle=True)
    test_dataset = configure_for_performance(
        test_dataset, batch_size=1, buffer_size=buffer_size, shuffle=False)
    return train_dataset, test_dataset



