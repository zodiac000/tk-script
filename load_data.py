import os
import re
import json
from collections import OrderedDict


def get_xys_from_file(file):
    with open(file, "r") as f:
        lines = f.readlines()
    coordinates = []
    for line in lines:
        coordinate = line.split(",")
        coordinates.append((int(coordinate[0]), int(coordinate[1])))
    return coordinates

def load_dictionary(image_loc, xy_loc):
    images = os.listdir(image_loc)
    images = list(filter(lambda x: re.search("\.jpg$", x), images)) 
    xys = get_xys_from_file(xy_loc)
    lens = len(images) if len(images) < len(xys) else len(xys)
    print("{} images.".format(str(len(images))))
    print("{} labels.".format(str(len(xys))))
    dicts = {}
    orderedDict = OrderedDict()
    for i in range(lens):
        dicts[os.path.join(image_loc, images[i])] = xys[i]

    keys = list(sorted(dicts.keys()))
    for key in keys:
        orderedDict[key] = dicts[key]

    with open('orderedDictionary.txt', 'w') as file:
        for key in orderedDict:
            file.write(key + "\t" + str(dicts[key][0]) + "," + str(dicts[key][1]) + "\n")
    
    # with open('dictionary.txt', 'w') as file:
        # file.write(json.dumps(dicts))
    return orderedDict

if __name__ == "__main__":
    image_loc = "/home/wenbin/Workspace/tk-script/images"
    xy_loc = "/home/wenbin/Workspace/tk-script/coordinates"
    dicts = load_dictionary(image_loc, xy_loc)
    for key, value in dicts.items():
        print("{}\t{}".format(key, value))
    print("="*30)
    for key, value in dicts.items():
        if value[0] == 595 and value[1] == 525:
            print("{}\t{}".format(key, value))

    print("{} number of records".format(len(dicts)))
