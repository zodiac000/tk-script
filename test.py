import tensorflow as tf
import numpy as np
from train import ds_test

ds_test = ds_test.take(3)
model = tf.keras.models.load_model('./check_points/mymodel.m5')
predictions = [model.predict(x) for x in ds_test]
