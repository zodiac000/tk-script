import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt 
from PIL import Image
from pdb import set_trace

matplotlib.use('tkagg')


titleTxt = """
threshold_cutout: {}
threshold_crop: {}
============================================
Mean Euclidean Distance {}
Number of Positive Predictions: {}
Number of Total Predictions: {}
"""


def cal_dist(l1, l2):
    return np.mean(np.sum((l1 - l2) ** 2, axis=1) ** 0.5)



# file_cutout = 'analysis_cutout.csv'
# file_crop = 'analysis_crop.csv'
# file_output = 'csv/pred_4615.csv'

# file_cutout = 'analysis_cutout_2.csv'
# file_crop = 'analysis_crop_2.csv'
# file_output = 'csv/pred_4615_2.csv'

file_cutout = 'analysis_cutout_50.csv'
file_crop = 'analysis_crop_50.csv'
file_output = 'csv/pred_6415_50.csv'

with open(file_cutout, 'r') as f:
    lines = f.readlines()
    lst_cutout = np.array([l.strip().split(',') for l in lines])

# with open('analysis_crop_2.csv', 'r') as f:
with open(file_crop, 'r') as f:
    lines = f.readlines()
    lst_crop = np.array([l.strip().split(',') for l in lines])



total = len(lst_cutout)
threshold_cutout = 1
# threshold_cutout = 0.5
# threshold_crop = 0
threshold_crop = 0.999

mask_1 = lst_cutout[:,5].astype(float) <= threshold_cutout
mask_2 = lst_crop[:, 5].astype(float) >= threshold_crop
valid = lst_cutout[mask_1 & mask_2]


names = valid[:, 0]
x = valid[:, 1].astype(int)
y = valid[:, 2].astype(int)
x_pred = valid[:, 3].astype(int)
y_pred = valid[:, 4].astype(int)
colors = valid[:, 5].astype(float)

dx = np.absolute(x - x_pred)
dy = np.absolute(y - y_pred)

label = valid[:, 1:3].astype(int)
pred = valid[:, 3:5].astype(int)
dist = cal_dist(label, pred)

abnormal = valid[dx > 200]

# for idx, ab in enumerate(abnormal):
    # x_label = ab[1].astype(int)
    # y_label = ab[2].astype(int)
    # x_pred = ab[3].astype(int)
    # y_pred = ab[4].astype(int)
    # image = Image.open('all_images/' + ab[0])
    # image_np = np.array(image)
    # image_np[y_label-3:y_label+3, x_label-3:x_label+3] = 255
    # image_np[y_pred-3:y_pred+3, x_pred-3:x_pred+3] = 0
    # image = Image.fromarray(image_np)
    # print(ab[0])
    # plt.title('label: {},{}    VS   prediction: {},{}'.format(x_label, y_label, x_pred, y_pred))
    # plt.imshow(image)
    # plt.show()

# with open('csv/pred_4615_2.csv', 'w') as f:
with open(file_output, 'w') as f:
    for i, pred in enumerate(valid):
        f.write(str(pred[0]) + ',' 
                + str(pred[3]) + ','
                + str(pred[4]) + '\n'
                )

print(len(valid))
print(dist)
# colors = np.random.rand(len(dx))
# colors = np.zeros(len(dx))
# colors = np.array([(lambda x:1 if x%2==0 else 0)(x) for x in range(len(dx))])

plt.scatter(dx, dy, c=colors)
# plt.scatter(dx, dy)
plt.title(titleTxt.format(threshold_cutout, threshold_crop, dist, len(valid), total))
plt.xlabel('E distance on X')
plt.ylabel('E distance on Y')
plt.show()

