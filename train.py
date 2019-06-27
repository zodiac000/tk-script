import tensorflow as tf
import pathlib
import random
from vgg16 import *

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
print("="*20)

# img_final = tf.image.resize_images(img_tensor, [192, 192])
# img_final = img_final/255.0
# print(img_final.shape)
# print(img_final.numpy().min())
# print(img_final.numpy().max())

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=1)
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

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_x, all_image_y))

def load_and_preprocess_from_path_label(path, x, y):
    return load_and_preprocess_image(path), x, y

image_label_ds = ds.map(load_and_preprocess_from_path_label)
print(image_label_ds)
ds_test = image_label_ds.take(1000)
ds_train = image_label_ds.skip(1000)

# import pdb
# pdb.set_trace()
BATCH_SIZE = 32
ds_train = ds_train.shuffle(buffer_size=image_count - 1000)
ds_train = ds_train.repeat()
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(buffer_size=AUTOTUNE)
print("{}\n ===============".format(ds))

ds_test = ds_test.shuffle(buffer_size=1000)
ds_test = ds_test.repeat()
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(buffer_size=AUTOTUNE)
print("{}\n ===============".format(ds))
# def change_range(image, x, y):
    # return 2*image-1, x, y
# keras_ds = ds.map(change_range)

# model = tf.keras.Sequential([
    # tf.keras.layers.Flatten(input_shape=(192,192,1)),
    # tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu),
    # tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu),
    # tf.keras.layers.Dense(1)
    # ])

# optimizer = tf.keras.optimizers.RMSprop(0.001)
# model.compile(loss=tf.losses.sigmoid_cross_entropy,
                # optimizer=optimizer,
                # metrics=["accuracy"])
# init = tf.global_variables_initializer()
# print(model.summary())




# fmap = np.zeros(shape=(7,7,1,2), dtype=np.float32)

# x = tf.placeholder(tf.float32, [None, 224, 224, 1], name='InputData')
# y = tf.placeholder(tf.float32, [None, 1], name='LabelData')

# feature_maps = tf.Variable(fmap)
# x = tf.nn.conv2d(x, feature_maps, strides=[1,1,1,1], padding="SAME") 
# y_pred = model(ds)

# merged_summary_op = tf.summary.merge_all()

graph = build_network(height=224, width=224, channel=1)
batch_size = 12
num_epochs = 50


tf.reset_default_graph()
with tf.Session() as sess:
    sess.run(init)

    writer = tf.summary.FileWriter('~/Workspace/tk-script/graphs', grapth=tf.get_default_graph())
    for epoch in range(10):
        avg_cost = 0
        total_batch = int(len(all_image_paths)/BATCH_SIZE)

        for i in range(total_batch):
            batch_input, batch_out_x, batch_out_y = ds_train.next_batch(BATCH_SIZE)
            c, summary = sess.run([cost, merged_summary_op], feed_dect={x:batch_input, y:[batch_out_x, batch_out_y]})

            writer.add_summary(summary, epoch * total_batch + i)
            avg_cost += c / total_batch
        # model.fit(ds_train, steps_per_epoch=10, validation_data=ds_test)



def train_network(graph, batch_size, num_epochs, pb_file_path):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        epoch_delta = 2
        for epoch_index in range(num_epochs):
            for i in range(12):
                sess.run([graph['optimize']], feed_dict={
                    graph['x']: np.reshape(x_train[i], (1, 224, 224, 3)),
                    graph['y']: ([[1, 0]] if y_train[i] == 0 else [[0, 1]])
                })
            if epoch_index % epoch_delta == 0:
                total_batches_in_train_set = 0
                total_correct_times_in_train_set = 0
                total_cost_in_train_set = 0.
                for i in range(12):
                    return_correct_times_in_batch = sess.run(graph['correct_times_in_batch'], feed_dict={
                        graph['x']: np.reshape(x_train[i], (1, 224, 224, 3)),
                        graph['y']: ([[1, 0]] if y_train[i] == 0 else [[0, 1]])
                    })
                    mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        graph['x']: np.reshape(x_train[i], (1, 224, 224, 3)),
                        graph['y']: ([[1, 0]] if y_train[i] == 0 else [[0, 1]])
                    })
                    total_batches_in_train_set += 1
                    total_correct_times_in_train_set += return_correct_times_in_batch
                    total_cost_in_train_set += (mean_cost_in_batch * batch_size)


                total_batches_in_test_set = 0
                total_correct_times_in_test_set = 0
                total_cost_in_test_set = 0.
                for i in range(3):
                    return_correct_times_in_batch = sess.run(graph['correct_times_in_batch'], feed_dict={
                        graph['x']: np.reshape(x_val[i], (1, 224, 224, 3)),
                        graph['y']: ([[1, 0]] if y_val[i] == 0 else [[0, 1]])
                    })
                    mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        graph['x']: np.reshape(x_val[i], (1, 224, 224, 3)),
                        graph['y']: ([[1, 0]] if y_val[i] == 0 else [[0, 1]])
                    })
                    total_batches_in_test_set += 1
                    total_correct_times_in_test_set += return_correct_times_in_batch
                    total_cost_in_test_set += (mean_cost_in_batch * batch_size)

                acy_on_test  = total_correct_times_in_test_set / float(total_batches_in_test_set * batch_size)
                acy_on_train = total_correct_times_in_train_set / float(total_batches_in_train_set * batch_size)
                print('Epoch - {:2d}, acy_on_test:{:6.2f}%({}/{}),loss_on_test:{:6.2f}, acy_on_train:{:6.2f}%({}/{}),loss_on_train:{:6.2f}'.format(epoch_index, acy_on_test*100.0,total_correct_times_in_test_set,
                                                                                                                                                   total_batches_in_test_set * batch_size,
                                                                                                                                                   total_cost_in_test_set,
                                                                                                                                                   acy_on_train * 100.0,
                                                                                                                                                   total_correct_times_in_train_set,
                                                                                                                                                   total_batches_in_train_set * batch_size,
                                                                                                                                                   total_cost_in_train_set))
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
            with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())

