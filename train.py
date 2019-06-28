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

image_ds_test = image_ds.take(1000)
image_ds_train = image_ds.skip(1000)

label_ds_test = label_ds.take(1000)
label_ds_train = label_ds.skip(1000)

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_x, all_image_y))

def load_and_preprocess_from_path_label(path, x, y):
    return load_and_preprocess_image(path), x, y

image_label_ds = ds.map(load_and_preprocess_from_path_label)

epochs = 10
batch_size = 64
iterations = len(all_image_paths) * epochs

# Generate the complete Dataset required in the pipeline
dataset = image_label_ds.repeat(epochs).batch(batch_size)
iterator = dataset.make_one_shot_iterator()

data_X, data_y_x, data_y_y = iterator.get_next()
data_y_x = tf.cast(data_y_x, tf.float32)
data_y_y = tf.cast(data_y_y, tf.float32)

# ds_test = image_label_ds.take(1000)
# ds_train = image_label_ds.skip(1000)
# BATCH_SIZE = 32
# ds_train = ds_train.shuffle(buffer_size=image_count - 1000)
# ds_train = ds_train.repeat()
# ds_train = ds_train.batch(BATCH_SIZE)
# ds_train = ds_train.prefetch(buffer_size=AUTOTUNE)
# print("{}\n ===============".format(ds))

# ds_test = ds_test.shuffle(buffer_size=1000)
# ds_test = ds_test.repeat()
# ds_test = ds_test.batch(BATCH_SIZE)
# ds_test = ds_test.prefetch(buffer_size=AUTOTUNE)
# print("{}\n ===============".format(ds))
init = tf.global_variables_initializer()
# print(model.summary())


graph = build_network(height=224, width=224, channel=1)
batch_size = 12
num_epochs = 50
optimize = graph['optimize']
cost = graph['cost']
x = graph['x']
y = graph['y']

display_epoch = 1
# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)


tf.reset_default_graph()
with tf.Session() as sess:
    # sess.run(init)

    writer = tf.summary.FileWriter('~/Workspace/tk-script/graphs', graph=tf.get_default_graph())
    for epoch in range(10):
        avg_cost = 0
        total_batch = int(len(all_image_paths)/BATCH_SIZE)
        for data in ds_train.__iter__():

        for i in range(total_batch):
            # batch_input, batch_out_x, batch_out_y = ds_train.next_batch(BATCH_SIZE)
            # c, opt = sess.run([cost, optimize], feed_dect={x:batch_input, y:[batch_out_x, batch_out_y]})
            c, opt = sess.run([cost, optimize], feed_dect={x:image_ds_train[i], y:label_ds_train[i]})

            writer.add_summary(c, epoch * total_batch + i)
            avg_cost += c / total_batch
        # model.fit(ds_train, steps_per_epoch=10, validation_data=ds_test)

        if (epoch+1) % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

# def train_network(graph, batch_size, num_epochs, pb_file_path):
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
        # sess.run(init)
        # epoch_delta = 2
        # for epoch_index in range(num_epochs):
            # for i in range(12):
                # sess.run([graph['optimize']], feed_dict={
                    # graph['x']: np.reshape(x_train[i], (1, 224, 224, 3)),
                    # graph['y']: ([[1, 0]] if y_train[i] == 0 else [[0, 1]])
                # })
            # if epoch_index % epoch_delta == 0:
                # total_batches_in_train_set = 0
                # total_correct_times_in_train_set = 0
                # total_cost_in_train_set = 0.
                # for i in range(12):
                    # return_correct_times_in_batch = sess.run(graph['correct_times_in_batch'], feed_dict={
                        # graph['x']: np.reshape(x_train[i], (1, 224, 224, 3)),
                        # graph['y']: ([[1, 0]] if y_train[i] == 0 else [[0, 1]])
                    # })
                    # mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        # graph['x']: np.reshape(x_train[i], (1, 224, 224, 3)),
                        # graph['y']: ([[1, 0]] if y_train[i] == 0 else [[0, 1]])
                    # })
                    # total_batches_in_train_set += 1
                    # total_correct_times_in_train_set += return_correct_times_in_batch
                    # total_cost_in_train_set += (mean_cost_in_batch * batch_size)


                # total_batches_in_test_set = 0
                # total_correct_times_in_test_set = 0
                # total_cost_in_test_set = 0.
                # for i in range(3):
                    # return_correct_times_in_batch = sess.run(graph['correct_times_in_batch'], feed_dict={
                        # graph['x']: np.reshape(x_val[i], (1, 224, 224, 3)),
                        # graph['y']: ([[1, 0]] if y_val[i] == 0 else [[0, 1]])
                    # })
                    # mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        # graph['x']: np.reshape(x_val[i], (1, 224, 224, 3)),
                        # graph['y']: ([[1, 0]] if y_val[i] == 0 else [[0, 1]])
                    # })
                    # total_batches_in_test_set += 1
                    # total_correct_times_in_test_set += return_correct_times_in_batch
                    # total_cost_in_test_set += (mean_cost_in_batch * batch_size)

                # acy_on_test  = total_correct_times_in_test_set / float(total_batches_in_test_set * batch_size)
                # acy_on_train = total_correct_times_in_train_set / float(total_batches_in_train_set * batch_size)
                # print('Epoch - {:2d}, acy_on_test:{:6.2f}%({}/{}),loss_on_test:{:6.2f}, acy_on_train:{:6.2f}%({}/{}),loss_on_train:{:6.2f}'.format(epoch_index, acy_on_test*100.0,total_correct_times_in_test_set,
                                                                                                                                                   # total_batches_in_test_set * batch_size,
                                                                                                                                                   # total_cost_in_test_set,
                                                                                                                                                   # acy_on_train * 100.0,
                                                                                                                                                   # total_correct_times_in_train_set,
                                                                                                                                                   # total_batches_in_train_set * batch_size,
                                                                                                                                                   # total_cost_in_train_set))
            # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
            # with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
                # f.write(constant_graph.SerializeToString())

