# import some common libraries
from sklearn.metrics import jaccard_score
from PIL import Image, ImageDraw
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
import random
import json
import cv2
import csv
import os

# import some common pytorch utilities
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset



BASE_DIR = '/localscratch/gna23/p4/CMPT_CV_lab4/'



'''
# This function should return a list of data samples in which each sample is a dictionary.
# Make sure to select the correct bbox_mode for the data
# For the test data, you only have access to the images, therefore, the annotations should be empty.
# Other values could be obtained from the image files.
# TODO: approx 35 lines
'''

def get_detection_data(set_name):
    img_base_dir = f'{BASE_DIR}/data/{set_name}'

    if set_name == "train":
        data_dirs = f'{BASE_DIR}/data/{set_name}.json'
        annotations = json.load(open(data_dirs))
        file_data = {}
        for idx, ech_ann in enumerate(tqdm(annotations)):
            filename = os.path.join(img_base_dir, ech_ann["file_name"])

            if filename not in file_data:
              width, height = Image.open(filename).size
              record = {}
              record['annotations'] = []
              record["file_name"] = filename
              record["image_id"] = idx
              record["height"] = height
              record["width"] = width
              file_data[filename] = record
            else:
              record = file_data[filename]


            ann = ech_ann['segmentation']
            px = [ann[0][i] for i in range(0, len(ann[0]), 2)]
            py = [ann[0][i] for i in range(1, len(ann[0]), 2)]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            obj = {"bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,"segmentation": [poly],"category_id": 0}
            record['annotations'].append(obj)


        data_dict = np.array(list(file_data.values()))
        N = len(data_dict)

        val_idx = np.random.choice(N, N//10)
        train_data = [data_dict[i] for i in range(N) if i not in val_idx]
        val_data = [data_dict[i] for i in range(N) if i in val_idx]

        return train_data,val_data

    elif set_name == "test":
        test_data = []
        for idx, filename in enumerate(tqdm(os.listdir(f'{BASE_DIR}/data/test'))):
            if ".ini" in filename:
              continue
            filename = os.path.join(img_base_dir, filename)
            width, height = Image.open(filename).size
            record = {}
            record['annotations'] = []
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
            test_data.append(record)
        return test_data

train_data,val_data = get_detection_data("train")
test_data = get_detection_data("test")


print(len(train_data))
print(len(val_data))
print(len(test_data))

data_collection = {
    "train":train_data,
    "validation":val_data,
    "test":test_data
}


def get_instance_sample(data, idx, img=None):

  if img==None:
    file_name = data['file_name']
    img = cv2.imread(file_name)

  # for idx in range(len(data['annotations'])):

  assert idx < len(data['annotations'])
  cur_ann = data['annotations'][idx]
  [tl_x,tl_y,br_x,br_y] = [int(i) for i in cur_ann['bbox']]
  mask = detectron2.utils.visualizer.GenericMask(cur_ann["segmentation"], data['height'], data['width']).mask

  sample_img = img[tl_y:br_y, tl_x:br_x]
  sample_mask = mask[tl_y:br_y, tl_x:br_x]*255

  obj_img = cv2.resize(sample_img, (128, 128), interpolation = cv2.INTER_AREA)
  obj_mask = cv2.resize(sample_mask, (128, 128), interpolation = cv2.INTER_AREA)

  return obj_img, obj_mask



'''
# We have provided a template data loader for your segmentation training
# You need to complete the __getitem__() function before running the code
# You may also need to add data augmentation or normalization in here
'''


class PlaneDataset(Dataset):
  def __init__(self, set_name, data_list):

      self.tran = transforms.Compose([
          transforms.ToTensor(), # Converting the image to tensor and change the image format (Channels-Last => Channels-First)
      ])

      self.set_name = set_name
      self.data = data_list
      self.instance_map = []
      for i, d in enumerate(self.data):
        for j in range(len(d['annotations'])):
          self.instance_map.append([i,j])

  '''
  # you can change the value of length to a small number like 10 for debugging of your training procedure and overfeating
  # make sure to use the correct length for the final training
  '''
  def __len__(self):
      return len(self.instance_map)

  def numpy_to_tensor(self, img, mask):
    if self.tran is not None:

        if random.random() > 0.5:
            img = cv2.flip(img,1)
            mask = cv2.flip(mask,1)
        if random.random() > 0.5:
            img = cv2.flip(img,0)
            mask = cv2.flip(mask,0)
        if random.random() > 0.3:
            brightness = random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            h, w = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

        img = self.tran(img)
        mask = self.tran(mask)
    img = torch.tensor(img, dtype=torch.float)
    mask = torch.tensor(mask, dtype=torch.float)
    return img, mask

  '''
  # Complete this part by using get_instance_sample function
  # make sure to resize the img and mask to a fixed size (for example 128*128)
  # you can use "interpolate" function of pytorch or "numpy.resize"
  # TODO: 5 lines
  '''
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()
    idx = self.instance_map[idx]
    data = self.data[idx[0]]

    img, mask = get_instance_sample(data,idx[1])
    img, mask = self.numpy_to_tensor(img, mask)

    return img, mask

def get_plane_dataset(set_name='train', batch_size=2):
    my_data_list = data_collection[set_name]
    dataset = PlaneDataset(set_name, my_data_list)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=32,
                                              pin_memory=True, shuffle=True)
    return loader, dataset



'''
# convolution module as a template layer consists of conv2d layer, batch normalization, and relu activation
'''
class conv(nn.Module):
    def __init__(self, in_ch, out_ch, activation=True):
        super(conv, self).__init__()
        if(activation):
          self.layer = nn.Sequential(
             nn.Conv2d(in_ch, out_ch, 3, padding=1),
             nn.BatchNorm2d(out_ch),
             nn.ReLU(inplace=True),
             nn.Conv2d(out_ch, out_ch, 3, padding=1),
             nn.BatchNorm2d(out_ch),
             nn.ReLU(inplace=True)
             )
        else:
          self.layer = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.layer(x)
        return x

'''
# downsampling module equal to a conv module followed by a max-pool layer
'''
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.layer = nn.Sequential(
            conv(in_ch, out_ch),
            nn.MaxPool2d(2)
            )

    def forward(self, x):
        x = self.layer(x)
        return x

'''
# upsampling module equal to a upsample function followed by a conv module
'''
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = conv(in_ch, out_ch,in_ch//2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
            self.conv = conv(in_ch, out_ch)

    def forward(self, x1,x2):
        x1 = self.up(x1)

        dY = x2.size()[2] - x1.size()[2]
        dX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [dX // 2, dX - dX // 2,
                        dY // 2, dY - dY // 2])
        x = torch.cat([x2, x1], dim=1) # adding skip connection here.

        y = self.conv(x)
        return y

'''
# the main model which you need to complete by using above modules.
# you can also modify the above modules in order to improve your results.
'''
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # Encoder

        self.input_conv = conv(3, 64)

        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)

        # Decoder

        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)

        self.output_conv = conv(64, 1, False) # ReLu activation is removed to keep the logits for the loss function


    def forward(self, input):
      y0 = self.input_conv(input)

      y1 = self.down1(y0)
      y2 = self.down2(y1)
      y3 = self.down3(y2)
      y4 = self.down4(y3)

      y = self.up1(y4,y3)
      y = self.up2(y,y2)
      y = self.up3(y,y1)
      y = self.up4(y,y0)
      output = self.output_conv(y)
      return output



'''
# The following is a basic training procedure to train the network
# You need to update the code to get the best performance
# TODO: approx ? lines
'''

# Training
# # Set the hyperparameters
# num_epochs = 50
# batch_size = 16
# learning_rate = 1e-3
# weight_decay = 1e-5
#
# model = MyModel() # initialize the model
# model = model.cuda() # move the model to GPU
# loader, _ = get_plane_dataset('train', batch_size) # initialize data_loader
# crit = nn.BCEWithLogitsLoss() # Define the loss function
# optim = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # Initialize the optimizer as SGD
#
# # start the training procedure
# for epoch in range(num_epochs):
#   total_loss = 0
#   for (img, mask) in tqdm(loader):
#     img = torch.tensor(img, device=torch.device('cuda'), requires_grad = True)
#     mask = torch.tensor(mask, device=torch.device('cuda'), requires_grad = True)
#     pred = model(img)
#     loss = crit(pred, mask)
#     optim.zero_grad()
#     loss.backward()
#     optim.step()
#     total_loss += loss.cpu().data
#   print("Epoch: {}, Loss: {}".format(epoch, total_loss/len(loader)))
#   torch.save(model.state_dict(), '{}/output/{}_segmentation_model.pth'.format(BASE_DIR, epoch))
#
# '''
# # Saving the final model
# '''
# torch.save(model.state_dict(), '{}/output/final_segmentation_model.pth'.format(BASE_DIR))


# eval

batch_size = 8
model = MyModel().cuda()
model.load_state_dict(torch.load('{}/output/final_segmentation_model.pth'.format(BASE_DIR)))
model = model.eval() # chaning the model to evaluation mode will fix the bachnorm layers
loader, dataset = get_plane_dataset('validation', batch_size)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def iou(gt, pd):
    gt = sigmoid(gt)
    pd = sigmoid(pd)
    intersection = np.count_nonzero(np.multiply(gt, pd))
    union = np.count_nonzero(gt + pd)
    if union != 0:
        return intersection / union
    else:
        return 0


total_iou = 0

ctr = 0
global_iou = 0
for (img, mask) in tqdm(loader):
    with torch.no_grad():
        img = img.cuda()
        mask = mask.cuda()
        mask = torch.unsqueeze(mask, 1).detach()
        pred = model(img).cpu().detach()
        for i in range(img.shape[0]):
            cur_pred = np.array(pred[i].cpu())[0]
            cur_mask = np.array(mask[i].cpu())[0]

            ctr += 1
            global_iou += iou(cur_mask, cur_pred)
print("\n #images: {}, Mean IoU: {}".format(ctr, global_iou/ctr))