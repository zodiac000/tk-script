import tensorflow as tf
import numpy as np
from get_data import get_ds
from model import create_model

_, ds_test = get_ds()
ds_test = ds_test.take(3)
model = create_model()
model.summary()
model.load_weights('check_points/mymodel.h5')
predictions = [model.predict(x) for x in ds_test]

for p in predictions:
    print(p)
