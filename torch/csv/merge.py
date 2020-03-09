import numpy as np
from pdb import set_trace

a1 = []
a2 = []
a3 = []
result = []
number_of_valid_data = 6415
with open ('pred_pass_valid_head_' + str(number_of_valid_data) + '_acc_x.csv', 'r') as f:
    lines = f.readlines()
    for line in lines:
        a1.append(line.strip().split(','))
    a1 = np.asarray(a1)

with open ('pred_pass_valid_head_' + str(number_of_valid_data) + '_acc_y.csv', 'r') as f:
    lines = f.readlines()
    for line in lines:
        a2.append(line.strip().split(','))
    a2 = np.asarray(a2)
    
with open('pred_pass_valid_head_' + str(number_of_valid_data) + '_adv.csv', 'r') as f:
    lines = f.readlines()
    for line in lines:
        a3.append(line.strip().split(','))
    a3 = np.asarray(a3)

for index, adv in enumerate(a3):
    # if adv[0] == a1[index][0]:
    result.append([a1[index][0], a1[index][1], a2[index][1], adv[1], adv[2], \
                    a1[index][3], a1[index][4]])

with open('merged_' + str(number_of_valid_data) + '.csv', 'w') as f:
    for line in result:
        f.write(','.join(line) + '\n')



