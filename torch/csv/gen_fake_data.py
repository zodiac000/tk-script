import numpy as np
from random import randint
from pdb import set_trace


def gen_neiboughs(data, radius, stride):
    temp = []
    for f, x, y in data:
        for i in range(-radius*stride, radius*stride+1, stride):
            for j in range(-radius*stride, radius*stride+1, stride):
                temp.append([f, x, y, int(x)+i, int(y)+j])

    return np.asarray(temp)


def gen_random():
    random_data = []
    for f, x, y in data:
        for i in range(50):
            random_data.append([f, x, y, randint(200, 800), randint(200, 800)])

    return np.asarray(random_data)





file_to_read = 'pass_valid_tail_1000.csv'
file_to_write = 'generated.csv'

with open(file_to_read, 'r') as f:
    lines = f.readlines()

data = []
for line in lines:
    data.append(line.strip().split(','))


data = np.asarray(data)

enlarged_data = gen_neiboughs(data, 3, 5)
random_data = gen_random()

combined_data = np.concatenate((enlarged_data, random_data), axis=0)


with open(file_to_write, 'w') as f:
    for data in combined_data:
        f.write(','.join(data)+'\n')

