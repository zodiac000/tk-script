with open("pass_images_7500.csv", 'r') as f:
    file_bc = f.readlines()

    for i, line in enumerate(file_bc):
        file_bc[i] = line.strip().split(',')


# print(file_bc)


with open("pass_ad.csv", 'r') as f:
    file_ad = f.readlines()

    for i, line in enumerate(file_ad):
        file_ad[i] = line.strip().split(',')


# print(file_ad)

ad_bc = []
for i, line_bc in enumerate(file_bc):
    temp = line_bc
    for j, line_ad in enumerate(file_ad):
        if line_bc[0] == line_ad[0]:
            temp.append(line_ad[1])
            temp.append(line_ad[2])
            break

    ad_bc.append(temp)



# print(ad_bc)
# from pdb import set_trace
# set_trace()
for i in ad_bc:
    if len(i) < 5:
        print(i)

    if i[2] == '-1' or i[4] == '-1':
        print(i)

# with open("pass_ad_bc_7500.csv", "w") as f:
    # for l in ad_bc:
        # f.write(",".join(l) + '\n')

