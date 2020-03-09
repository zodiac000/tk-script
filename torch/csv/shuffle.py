import numpy as np
from numpy.random import shuffle

with open('pass_valid_1000+79.csv', 'r') as f_read:
    lines = f_read.readlines()
    shuffle(lines)
    with open('pass_valid_1000+79_shuffle_train.csv', 'w') as f_write:
        for line in lines[:1000]:
            f_write.write(line)
    with open('pass_valid_1000+79_shuffle_valid.csv', 'w') as f_write:
        for line in lines[1000:]:
            f_write.write(line)
