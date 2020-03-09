import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from pdb import set_trace

matplotlib.use('tkagg')
lst = []
file_name = 'pred_pass_valid_7000_acc.csv'
with open(file_name, 'r') as f:
    lines = f.readlines()
    for line in lines:
        items = line.strip().split(',')
        if float(items[2]) > 0.15:
            lst.append(items)

lst = np.asarray(lst)

print('number of predictions is {}'.format(lst.shape[0]))
print('mean distance of valid prediction is {}'.format(\
                lst[:,3].astype(float).mean()))


conf_x = lst[:, 1].astype(float)
conf_y = lst[:, 2].astype(float)
dist = lst[:, 3].astype(float)
fig = plt.figure()

#   3-D
ax = Axes3D(fig)
ax.scatter(conf_x, conf_y, dist)
ax.set_xlabel('conf_x')
ax.set_ylabel('conf_y')
ax.set_zlabel('distance')


#  2-D
# ax = fig.add_axes([0.1,0.1,0.9,0.9])
# ax.scatter(conf_x, dist, color='r')
# ax.set_xlabel('confidence')
# ax.set_ylabel('euclidean_distance')



plt.show()

# with open('valid.csv', 'w') as f:
    # for line in lst:
        # f.write(','.join(line[:3]) + '\n')
