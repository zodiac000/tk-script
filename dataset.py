import pandas as pd
import tensorflow as tf
import random as rd
import numpy as np
import pdb
import cv2
import numpy as np

shape = 224
width = 1280.0
height = 1024.0

def translate(image):
    translations = (np.random.rand(2) - 0.5) * [width / 100, height / 100]
    translations = [int(value) for value in translations]
    return tf.contrib.image.translate(image, translations, 'BILINEAR'), translations

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    # image = tf.image.random_brightness(image, 0.05)
    # image = tf.image.random_contrast(image, 0.7, 1.3)
    image = tf.to_float(image)
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_from_path_label_translate(path, x, y):
    image = load_and_preprocess_image(path)
    image, translations = translate(image)
    image = tf.image.resize(image, [shape, shape])
    return image, tf.stack([tf.to_float((x+translations[0]))/width, tf.to_float((y+translations[1]))/height])

def load_and_preprocess_from_path_label(path, x, y):
    image = load_and_preprocess_image(path)
    image = tf.image.resize(image, [shape, shape])
    return image, tf.stack([tf.to_float(x)/width, tf.to_float(y)/height])

def get_ds(dict_loc, epochs, batch_size):
    all_image_paths, all_x, all_y = get_data_list(dict_loc)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_image_count = len(all_image_paths[0])
    print('train_image_count: {}'.format(train_image_count))
    
    train_image_label_ds = tf.data.Dataset.from_tensor_slices((all_image_paths[0], all_x[0], all_y[0]))
    # train_image_label_ds = train_image_label_ds.map(load_and_preprocess_from_path_label_translate)
    train_image_label_ds = train_image_label_ds.map(load_and_preprocess_from_path_label)
    test_image_label_ds = tf.data.Dataset.from_tensor_slices((all_image_paths[1], all_x[1], all_y[1]))
    test_image_label_ds = test_image_label_ds.map(load_and_preprocess_from_path_label)

    ds_train = train_image_label_ds.shuffle(buffer_size=train_image_count) \
                                   .repeat(epochs) \
                                   .batch(batch_size) \
                                   .prefetch(buffer_size=AUTOTUNE)

    ds_test = test_image_label_ds.batch(batch_size) \
                            .prefetch(buffer_size=AUTOTUNE)

    return ds_train, ds_test 


def get_data_list(dict_loc):
    data = pd.read_csv(dict_loc, names=['path', 'x', 'y'])
    all_image_paths = [data.path.tolist()[1000:], data.path.tolist()[:1000]]
    all_x = [data.x.tolist()[1000:], data.x.tolist()[:1000]]
    all_y = [data.y.tolist()[1000:], data.y.tolist()[:1000]]
    
    return all_image_paths, all_x, all_y



if __name__ == "__main__":
    ds_train, ds_test = get_ds()
