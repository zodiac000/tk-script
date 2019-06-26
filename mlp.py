import tensorflow as tf
import pathlib
import random


# tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

# data_root = '/home/wenbin/Workspace/tk-script/images'
# data_root = pathlib.Path(data_root)
# all_image_paths = list(data_root.glob('*.jpg'))
# all_image_paths = [str(path) for path in all_image_paths]

# random.shuffle(all_image_paths)
# print(len(all_image_paths))

dict_loc = '/home/wenbin/Workspace/tk-script/saved_dict.txt'
with open(dict_loc, 'r') as file:
    lines = file.readlines()
dicts = {}
for line in lines:
    splited = line.split("\t")
    xy = splited[1].split(",")
    dicts[splited[0]] = (int(xy[0]), int(xy[1]))
xys = []
directs = []
for directory in dicts.keys():
    xys.append(dicts[directory])
    directs.append(directory)
all_image_paths = directs
all_image_x = [xy[0] / 1280 for xy in xys]
all_image_y = [xy[1] / 1024 for xy in xys]
image_count = len(all_image_paths)
print(len(all_image_paths))

img_path = all_image_paths[0]
print(img_path)
img_raw = tf.read_file(img_path)
print(repr(img_raw)[:100]+"...")

img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.shape)
print(img_tensor.dtype)

# img_final = tf.image.resize_images(img_tensor, [192, 192])
# img_final = img_final/255.0
# print(img_final.shape)
# print(img_final.numpy().min())
# print(img_final.numpy().max())

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize_images(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
print(type(path_ds))
print(path_ds)

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_x, tf.float32))

# for label in label_ds.take(3):
    # print(label)

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_x))

def load_and_preprocess_from_path_label(path, x):
    return load_and_preprocess_image(path), x

image_label_ds = ds.map(load_and_preprocess_from_path_label)
print(image_label_ds)

BATCH_SIZE = 32
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
print(ds)

# def change_range(image, x, y):
    # return 2*image-1, x, y
# keras_ds = ds.map(change_range)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(192,192,1)),
    tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(1)
    ])
# y_pred = model(ds)
optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss=tf.losses.mean_squared_error,
                optimizer=optimizer,
                metrics=["accuracy"])
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(10):
        model.fit(ds, steps_per_epoch=10)

