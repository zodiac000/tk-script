import pandas as pd
import tensorflow as tf
import random as rd
import tensorflow as tf
import numpy as np
import pdb

# input_size = 324
input_size = 512
width = 1280.0
height = 1024.0


def generate_random_bbox():
    return int(rd.random() * (width - input_size)), int(rd.random() * (height - input_size))


def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  image = tf.image.decode_jpeg(image, channels=1)
  # image = tf.image.random_brightness(image, 0.05)
  # image = tf.image.random_contrast(image, 0.7, 1.3)
  # image = tf.image.resize(image, [input_size, input_size])

  x, y = generate_random_bbox()
  image = tf.image.crop_to_bounding_box(image, x, y, input_size, input_size)
  # image /= 255.0  # normalize to [0,1] range
  image = tf.cast(image, tf.float32) / 255.0
  return image, x, y

# def load_and_preprocess_from_path_label(path, x, y):
    # return load_and_preprocess_image(path), tf.stack([x, y])

def get_ds(dict_loc, epochs, batch_size):
    all_image_paths, all_x, all_y = get_data(dict_loc)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    image_count = len(all_image_paths)
    print('image_count: {}'.format(image_count))
    image_label_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, 
                                                         tf.cast(all_x, tf.float32),
                                                         tf.cast(all_y, tf.float32)))


    # image_label_ds = image_label_ds.map(load_and_preprocess_from_path_label)
    for index, (path, x, y) in enumerate(image_label_ds):
        image, new_x, new_y = load_and_preprocess_image(path)
        x = x.numpy()
        y = y.numpy()
        x = tf.cast(x - new_x, tf.float32)
        y = tf.cast(y - new_y, tf.float32)
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)
        xy = tf.stack([x, y])
        pdb.set_trace()
        image_label_ds[index] = (image, xy)

    print('image_count: {}'.format(image_count))
    ds_train = image_label_ds.skip(1000).shuffle(buffer_size=image_count - 1000) \
                             .repeat(epochs) \
                             .batch(batch_size) \
                             .prefetch(buffer_size=AUTOTUNE)

    ds_test = image_label_ds.take(1000) \
                            .batch(batch_size) \
                            .prefetch(buffer_size=AUTOTUNE)
    return ds_train, ds_test 

def get_data(dict_loc):
    data = pd.read_csv(dict_loc, names=['path', 'x', 'y'])
    all_image_paths = data.path.tolist()
    all_x = [int(x)/width for x in data.x.tolist()]
    all_y = [int(y)/height for y in data.y.tolist()]
    
    return all_image_paths, all_x, all_y

if __name__ == "__main__":
    ds_train, ds_test = get_ds()
