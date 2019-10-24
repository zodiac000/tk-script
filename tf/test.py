import tensorflow as tf
import numpy as np
from dataset import get_ds, get_data_list
from model import create_model


pred_dict = "pred_dict.csv"
saved_dict = 'saved_dict.csv'
batch_size = 100


_, ds_test = get_ds(saved_dict, 1, batch_size)
ds_test = ds_test.take(1)

model = create_model()
model.summary()
model.load_weights('check_points/mymodel_48.h5')

all_image_paths, all_x, all_y = get_data_list(saved_dict)
all_image_paths = all_image_paths[1]
all_x = [x / 1280.0 for x in all_x[1]]
all_y = [y / 1024.0 for y in all_y[1]]

# import pdb
# pdb.set_trace()

# predictions = [model.predict(x) for x in ds_test]
predictions = model.predict(ds_test)

def save_xy():
    with open(pred_dict, "w") as f:
        for i, pred in enumerate(predictions):
            f.write(all_image_paths[i] + "," + str(int(pred[0] * 1280)) + "," + str(int(pred[1] * 1024)) + "\n")
    print("write {} predictions to {}".format(batch_size, pred_dict))

for index, pred in enumerate(predictions):
    print("{}: predicted: ({}, {})\tlabel: ({}, {})".format(all_image_paths[index], pred[0], pred[1], all_x[index], all_y[index]))
save_xy()
