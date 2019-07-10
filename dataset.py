import pandas as pd
import tensorflow as tf
import random as rd
import numpy as np
import pdb
import cv2
import numpy as np

input_size = 224
width = 1280.0
height = 1024.0


def generate_random_bbox():
    return float(int(rd.random() * (width - input_size))), float(int(rd.random() * (height - input_size)))


def load_and_preprocess_from_path_label(path, x, y, flag):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    # image = tf.image.random_brightness(image, 0.05)
    # image = tf.image.random_contrast(image, 0.7, 1.3)
    # image = tf.image.resize(image, [input_size, input_size])

    x_delta, y_delta = generate_random_bbox()
    image = tf.image.crop_to_bounding_box(image, int(x_delta), int(y_delta), input_size, input_size)
    image = tf.cast(image, tf.float32) / 255.0
    x = x.numpy() - x_delta
    if x < 0:
        flag = tf.convert_to_tensor(0.0)
    x = tf.convert_to_tensor(x)
    # pdb.set_trace() 
    # x_new = x - x_delta
    # print(x)
    # print(type(x))
    # print(tf.math.equal(x, tf.convert_to_tensor(x_delta)))
    # with tf.compat.v1.Session() as sess:
        # print(x.eval())
        # # x = x.eval(session=sess)
        # print(x)
    # print("="*20)
    # # print(x[])
    # aaa = x - tf.convert_to_tensor(x_delta)
    # print(aaa)
    # print(type(aaa))
    # flag = 0 if (x-x_delta < 0 or y-y_delta < 0) else 1
    return image, tf.stack([x, y, flag, x_delta, y_delta])

def get_ds(dict_loc, epochs, batch_size):
    all_image_paths, all_x, all_y, all_flag = get_data(dict_loc)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    image_count = len(all_image_paths)
    print('image_count: {}'.format(image_count))

    # pdb.set_trace()
    image_label_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, 
                                                         tf.cast(all_x, tf.float32),
                                                         tf.cast(all_y, tf.float32),
                                                         tf.cast(all_flag, tf.float32)))


shape = 224
width = 1280.0
height = 1024.0

def translate(image):
    translations = np.random.rand(2) * [width - shape, height - shape]
    translations = [int(value) for value in translations]
    return tf.contrib.image.translate(image, translations, 'BILINEAR'), translations



def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    # image = tf.image.random_brightness(image, 0.05)
    # image = tf.image.random_contrast(image, 0.7, 1.3)
    # image = tf.image.resize(image, [shape, shape])
    image = tf.to_float(image)
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_from_path_label(path, x, y):
    image = load_and_preprocess_image(path)
    image, translations = translate(image)
    image = tf.image.resize(image, [shape, shape])
    # image = tf.to_float(image)
    return image, tf.stack([tf.to_float(x-translations[0])/width, tf.to_float(y-translations[1])/height])
    # return image, tf.stack([x, y])

def get_ds(dict_loc, epochs, batch_size):
    all_image_paths, all_x, all_y = get_data_list(dict_loc)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    image_count = len(all_image_paths)
    print('image_count: {}'.format(image_count))
    image_label_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_x, all_y))
                                                         # tf.cast(all_x, tf.float32),
                                                         # tf.cast(all_y, tf.float32)))


    image_label_ds = image_label_ds.map(load_and_preprocess_from_path_label)

    print('image_count: {}'.format(image_count))
    # ds_train = image_label_ds.skip(1000).shuffle(buffer_size=image_count - 1000) \
                             # .repeat(epochs) \
                             # .batch(batch_size) \
                             # .prefetch(buffer_size=AUTOTUNE)
    ds_train = image_label_ds.skip(1000) \
                             .repeat(epochs) \
                             .batch(batch_size) \
                             .prefetch(buffer_size=AUTOTUNE)

    ds_test = image_label_ds.take(1000) \
                            .batch(batch_size) \
                            .prefetch(buffer_size=AUTOTUNE)
    return ds_train, ds_test 

# def get_data(dict_loc):
    # data = pd.read_csv(dict_loc, names=['path', 'x', 'y', 'flag'])
    # all_image_paths = data.path.tolist()
    # all_x = [int(x)/width for x in data.x.tolist()]
    # all_y = [int(y)/height for y in data.y.tolist()]
    # all_flag = [int(flag) for flag in data.flag.tolist()]


def get_data_list(dict_loc):
    data = pd.read_csv(dict_loc, names=['path', 'x', 'y'])
    all_image_paths = data.path.tolist()
    all_x = data.x.tolist()
    all_y = data.y.tolist()
    # all_x = [int(x)/width for x in data.x.tolist()]
    # all_y = [int(y)/height for y in data.y.tolist()]
    
    return all_image_paths, all_x, all_y, all_flag



if __name__ == "__main__":
    ds_train, ds_test = get_ds()
