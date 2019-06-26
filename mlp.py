import tensorflow as tf
import pathlib
import random


tf.enable_eager_execution()
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
all_image_labels = [xy[0] for xy in xys]
all_image_x = [xy[0] for xy in xys]
all_image_y = [xy[1] for xy in xys]
image_count = len(all_image_paths)
print(len(all_image_paths))
print(len(all_image_labels))

img_path = all_image_paths[0]
print(img_path)
img_raw = tf.read_file(img_path)
print(repr(img_raw)[:100]+"...")

img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.shape)
print(img_tensor.dtype)

img_final = tf.image.resize_images(img_tensor, [192, 192])
img_final = img_final/255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())

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

# import matplotlib.pyplot as plt

# plt.figure(figsize=(8,8))
# for n,image in enumerate(image_ds.take(4)):
  # plt.subplot(2,2,n+1)
  # plt.imshow(image)
  # plt.grid(False)
  # plt.xticks([])
  # plt.yticks([])

# plt.show()

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

for label in label_ds.take(3):
    print(label)

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_x, all_image_y))

def load_and_preprocess_from_path_label(path, x, y):
    return load_and_preprocess_image(path), x, y

image_label_ds = ds.map(load_and_preprocess_from_path_label)
print(image_label_ds)

BATCH_SIZE = 32
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
print(ds)
