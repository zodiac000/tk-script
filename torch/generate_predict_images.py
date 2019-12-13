import os
from PIL import Image
import numpy as np
from pdb import set_trace
from tqdm import tqdm

def predict(csv, image_dir, save_dir):
    with open(csv, 'r') as file:
        lines = file.readlines()

    pbar = tqdm(total = len(lines))
    for line in lines:
        name, x, y = line.split(',')
        x, y = int(x), int(y)
        image_path = os.path.join(image_dir, name)
        image = Image.open(image_path) 
        image = np.asarray(image)
        rgb_image = np.concatenate((image[...,np.newaxis], \
                                    image[..., np.newaxis], \
                                    image[..., np.newaxis]), axis=2)
        for i in range(-2, 3):
            for j in range(-2, 3):
                rgb_image[y+i,x+j,0] = 255
                rgb_image[y+i,x+j,1] = 0
                rgb_image[y+i,x+j,2] = 0
                
        color_image = Image.fromarray(np.uint8(rgb_image), 'RGB')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        color_image.save(os.path.join(save_dir, name))
        pbar.update()


if __name__ == "__main__":
    # csv = 'csv/pred_dict_cascade4_6500.csv'
    csv = 'csv/pred_failed_cascade4_6500_100.csv'
    image_dir = 'failed_images'
    save_dir = 'pred_fail'
    predict(csv, image_dir, save_dir)
