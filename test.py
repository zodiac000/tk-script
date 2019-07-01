import tensorflow as tf
import numpy as np
from dataset import get_ds, get_data
from model import create_model


batch_size = 32
pred_dict = "pred_dict.csv"
dict_loc = 'saved_dict.csv'

_, ds_test = get_ds(dict_loc, batch_size, 1)
ds_test = ds_test.take(1)

model = create_model()
model.summary()
model.load_weights('check_points/mymodel_8.h5')

all_image_paths, all_x, all_y = get_data(pred_dict)
predictions = [model.predict(x) for x in ds_test]


def save_xy():
    with open(pred_dict, "w") as f:
        for i, pred in enumerate(predictions[0]):
            f.write(all_image_paths[i] + "," + str(int(pred[0] * 1280)) + "," + str(int(pred[1] * 1024)) + "\n")
    print("write {} predictions to {}".format(batch_size, pred_dict))


for index, pred in enumerate(predictions[0]):
    print("{}: predicted: ({}, {})\tlabel: ({}, {})".format(all_image_paths[index], pred[0], pred[1], all_x[index], all_y[index]))
save_xy()
