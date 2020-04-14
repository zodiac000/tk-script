from Classifier_cutout import Classifier
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset import CropDataset, CropDataset_pred
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import math
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image
from random import uniform
from utils import heatmap_to_coor, accuracy_sum, spike, gaussion_hmap
from tqdm import tqdm

from pdb import set_trace



#Evaluate on test_cutout_cls model
# def eval_cls():
    # correct_real = 0
    # correct_fake = 0
    # correct_rand_cut = 0
    # correct_rand_center_cut = 0
    # total_real = 0
    # total_fake = 0
    # total_rand_cut = 0
    # total_rand_center_cut = 0

    # batch_size = 10
    # invalid_batch_size = 5
    # train_csv = './csv/pass_valid_test_100.csv'
    # dir_weight = 'check_points/classifier_cutout.pth'
    # dataset = CropDataset(csv_file=train_csv)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # classifier = Classifier().cuda()
    # classifier.load_state_dict(torch.load(dir_weight))
    # classifier.eval()

    # with torch.no_grad():
        # for i, batch_data in enumerate(dataloader):
            # image_real = batch_data['image'].cuda()
            # image_fake = batch_data['coor_crop'].cuda()
            # image_valid_rand_cuts = batch_data['random_cutouts'].cuda().permute(1,0,2,3,4)
            # image_valid_rand_center_cuts = batch_data['random_center_cutouts'].cuda().permute(1,0,2,3,4)

            # b_size_real = len(image_real)
            # b_size_fake = len(image_fake)
            # b_valid_rand_cut_size = image_valid_rand_cuts.shape[0] * image_valid_rand_cuts.shape[1]
            # b_valid_rand_center_cut_size = image_valid_rand_center_cuts.shape[0] * image_valid_rand_center_cuts.shape[1]

            # logits_real = classifier(image_real)
            # logits_fake = classifier(image_fake)

            # for idx, image in enumerate(image_valid_rand_cuts):
                # logits_valid_rand_cut = classifier(image)
                # correct_rand_cut += (logits_valid_rand_cut>=0.5).sum().item() 
        
            # for idx, image in enumerate(image_valid_rand_center_cuts):
                # logits_valid_rand_center_cut = classifier(image)
                # correct_rand_center_cut += (logits_valid_rand_center_cut<0.5).sum().item() 

            # correct_real += (logits_real>=0.5).sum().item()
            # correct_fake += (logits_fake<0.5).sum().item()

            # total_real += b_size_real
            # total_fake += b_size_fake
            # total_rand_cut += b_valid_rand_cut_size
            # total_rand_center_cut += b_valid_rand_center_cut_size

        # print('acc real: {}/{}\tacc fake: {}/{}\tacc rand_cut: {}/{}\tacc_rand_center_cut: {}/{}'.format(
                # correct_real,
                # total_real,
                # correct_fake,
                # total_fake,
                # correct_rand_cut,
                # total_rand_cut,
                # correct_rand_center_cut,
                # total_rand_center_cut,
                # ))

#Test on predictions
def eval_prediction():
    batch_size = 100
    invalid_batch_size = 5
    # train_csv = './csv/pred_pass_valid_test_100.csv'
    train_csv = './csv/pred_pass_valid_head_6415.csv'
    # train_csv = './csv/1000pred.csv'
    dir_weight = 'check_points/classifier_crop_50.pth'
    dataset = CropDataset_pred(csv_file=train_csv)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    classifier = Classifier().cuda()
    classifier.load_state_dict(torch.load(dir_weight))
    classifier.eval()
    
    with torch.no_grad():
        predictions = []
        all_logits = []
        all_labels = []
        all_preds = []
        all_names = []
        pbar = tqdm(total=len(dataloader))
        # positive = 0
        # total = 0
        for i, batch_data in enumerate(dataloader):
            # print('{}/{}'.format(i, len(dataloader)))
            coor_crop = batch_data['coor_crop'].cuda()
            image_name = batch_data['image_name']
            coor_label = batch_data['coor_label']
            coor_pred = batch_data['coor_pred']
            b_size = len(coor_crop)
            logits = classifier(coor_crop)
            for l in logits:
                all_logits.append(l.flatten().cpu().detach().item())
            # threshold = 0.0005
            # positive += (logits<threshold).sum().item()
            # total += b_size
            # results = [0 if l.item()<threshold else 1 for l in logits]
            # predictions.extend(results)
            all_labels.extend(coor_label.numpy())
            all_preds.extend(coor_pred.numpy())
            all_names.extend(image_name)

            pbar.update()
        # print('positive / total: {}/{}'.format(
                # positive,
                # total,
                # ))
    with open('analysis_crop_50.csv', 'w') as f:
        for i, name in enumerate(all_names):
            f.write(name + ',' \
                    + str(all_labels[i][0]) + ',' 
                    + str(all_labels[i][1]) + ',' 
                    + str(all_preds[i][0]) + ',' 
                    + str(all_preds[i][1]) + ',' 
                    + str(all_logits[i]) 
                    # + str(predictions[i]) 
                    + '\n')

if __name__ == '__main__':
    eval_prediction()

