import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from pdb import set_trace
import matplotlib

matplotlib.use('tkagg')


lst = []


with open('merged_6415.csv', 'r') as f:
    lines = f.readlines()
    for line in lines:
        items = line.strip().split(',')
        # if float(items[2]) > 0.000915 and float(items[3]) > 0.01 :
            # lst.append(items)
        lst.append(items[:-2])


lst = np.asarray(lst)
x_min = lst[:,1].astype(float).min()
x_max = lst[:,1].astype(float).max()
y_min = lst[:,2].astype(float).min()
y_max = lst[:,2].astype(float).max()

all_valid_points = []
low_points = np.empty((0,5))
high_points = np.empty((0,5))
threshold = 0.78
threshold_x = (x_max - x_min) * threshold + x_min
threshold_y = (y_max - y_min) * threshold + y_min
for i in lst:
    # if float(i[1]) < threshold_x and float(i[2]) < threshold_y and float(i[3]) > 0.1:
        all_valid_points.append(i)
        if float(i[4]) <= 10.0:
            low_points = np.append(low_points, [i], axis=0)
        else:
            high_points = np.append(high_points, [i], axis=0)
all_valid_points = np.array(all_valid_points)

print('range of x is {} ----- {}\tthreshold_x is {}'.format(x_min, x_max, threshold_x))
print('range of y is {} ----- {}\tthreshold_y is {}'.format(y_min, y_max, threshold_y))


print('number of valid prediction is {}'.format(len(all_valid_points)))
print('mean distance of valid prediction is {}'.format(\
                all_valid_points[:,4].astype(float).mean()))



#     2-D
# conf = lst[:, 1].astype(float)
# dist = lst[:, 2].astype(float)
# fig = plt.figure()
# plt.scatter(conf, dist, c='g')
# plt.xlabel('conf')
# plt.ylabel('dist')


#     3-D
low_conf_x = low_points[:, 2].astype(float)
low_conf_y = low_points[:, 3].astype(float)
low_dist = low_points[:, 4].astype(float)

high_conf_x = high_points[:, 2].astype(float)
high_conf_y = high_points[:, 3].astype(float)
high_dist = high_points[:, 4].astype(float)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(low_conf_x, low_conf_y, low_dist, c='b')
ax.scatter(high_conf_x, high_conf_y, high_dist, c='r')
ax.set_xlabel('conf_x_y')
ax.set_ylabel('conf_adv')
ax.set_zlabel('dist')


with open('filtered_6415.csv', 'w') as f:
    for line in lst:
        temp = line[0:1].tolist() + line[5:].tolist()
        f.write(','.join(temp) + '\n')


plt.show()



