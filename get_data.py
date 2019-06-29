import pandas as pd
import tensorflow as tf

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  image = tf.image.decode_jpeg(image, channels=1)
  image = tf.image.resize(image, [224, 224])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_from_path_label(path, x, y):
    return load_and_preprocess_image(path), tf.stack([x, y])

def get_ds():
    dict_loc = '/home/wenbin/Workspace/tk-script/saved_dict.csv'
    data = pd.read_csv(dict_loc, names=['path', 'x', 'y'])
    all_image_paths = data.path.tolist()
    all_x = [int(x)/1280.0 for x in data.x.tolist()]
    all_y = [int(y)/1024.0 for y in data.y.tolist()]
    image_count = len(all_image_paths)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    image_label_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, 
                                                         tf.cast(all_x, tf.float32),
                                                         tf.cast(all_y, tf.float32)))


    image_label_ds = image_label_ds.map(load_and_preprocess_from_path_label)

    batch_size = 32
    epochs = 10

    ds_train = image_label_ds.skip(1000).shuffle(buffer_size=image_count) \
                             .repeat(epochs) \
                             .batch(batch_size) \
                             .prefetch(buffer_size=AUTOTUNE)

    ds_test = image_label_ds.take(1000) \
                            .batch(batch_size) \
                            .prefetch(buffer_size=AUTOTUNE)
    return ds_train, ds_test


if __name__ == "__main__":
    ds_train, ds_test = get_ds()
