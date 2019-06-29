import tensorflow as tf
import pathlib
import random
from vgg16 import *
from tqdm import tqdm
import pandas as pd
# tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

dict_loc = '/home/wenbin/Workspace/tk-script/saved_dict.csv'
data = pd.read_csv(dict_loc, names=['path', 'x', 'y'])
all_image_paths = data.path.tolist()
all_x = [int(x)/1280.0 for x in data.x.tolist()]
all_y = [int(y)/1024.0 for y in data.y.tolist()]
image_count = len(all_image_paths)


def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  image = tf.image.decode_jpeg(image, channels=1)
  image = tf.image.resize(image, [224, 224])
  image /= 255.0  # normalize to [0,1] range
  return image

image_label_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, 
                                                     tf.cast(all_x, tf.float32),
                                                     tf.cast(all_y, tf.float32)))

def load_and_preprocess_from_path_label(path, x, y):
    return load_and_preprocess_image(path), tf.stack([x, y])

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

from tensorflow.keras import datasets, layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mean_squared_error'])

tensorboard_cb = tf.keras.callbacks.TensorBoard(update_freq='batch')
callbacks = [
    tensorboard_cb,
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./check_points/mymodel_{epoch}.h5',
        save_best_only=True,
        monitor='val_loss',
        verbose=1)
]
model.fit(ds_train, epochs=5, validation_data=ds_test, callbacks=callbacks)

