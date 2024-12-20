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
import warnings


warnings.filterwarnings("ignore")


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

    if set_name != "test":
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
        train_data = train_data+val_data
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

# train_data,val_data = get_detection_data("train")
# test_data = get_detection_data("test")


# print(len(train_data))
# print(len(val_data))
# print(len(test_data))


TRAIN_DETECTION = False
EVAL_DETECTION = False

TRAIN_SEGMENTATION = True
EVAL_SEGMENTATION = True

GEN_CSV = True
RCNN_TRAIN = False
RCNN_EVAL = False

def normalize_image(img):
    img = img.to(dtype=torch.float32)
    min_val = img.min()
    max_val = img.max()
    normalized = 255 * (img - min_val) / (max_val - min_val)
    return normalized.to(dtype=torch.uint8)

def is_gpu_available():
  return torch.cuda.is_available()

def get_seg_model(pth='{}/output_v3/final_segmentation_model.pth'.format(BASE_DIR)):
  if is_gpu_available():
    model = MyModel().cuda()
    model.load_state_dict(torch.load(pth))
  else:
    model = MyModel()
    model.load_state_dict(torch.load(pth,map_location='cpu'))
  return model

# DatasetCatalog.remove('data_detection_train')
# MetadataCatalog.remove('data_detection_train')
#
# DatasetCatalog.remove('data_detection_validation')
# MetadataCatalog.remove('data_detection_validation')
#
#
# DatasetCatalog.remove('data_detection_test')
# MetadataCatalog.remove('data_detection_test')

for sel in ["merge","test"]:
    if sel == "merge":
        temp_train, temp_validation = get_detection_data(sel)
        DatasetCatalog.register("data_detection_train", lambda sel=sel: temp_train)
        MetadataCatalog.get("data_detection_train").set(thing_classes=["plane"])

        DatasetCatalog.register("data_detection_validation", lambda sel=sel: temp_validation)
        MetadataCatalog.get("data_detection_validation").set(thing_classes=["plane"])
    else:
        DatasetCatalog.register("data_detection_" + sel, lambda sel=sel: get_detection_data(sel))
        MetadataCatalog.get("data_detection_" + sel).set(thing_classes=["plane"])

train_metadata = MetadataCatalog.get("data_detection_train")
validation_metadata = MetadataCatalog.get("data_detection_validation")
test_metadata = MetadataCatalog.get("data_detection_test")


cfg = get_cfg()
cfg.OUTPUT_DIR = "{}/output/".format(BASE_DIR)
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.SOLVER.MAX_ITER = 10000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1



cfg.DATASETS.TRAIN = ("data_detection_train",)
cfg.DATASETS.TEST = ("data_detection_test",)


cfg.OUTPUT_DIR = f"{BASE_DIR}/output/"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)




from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data import build_detection_train_loader

import copy
def collate_fn(ech_data):
    ech_data = copy.deepcopy(ech_data)
    image = utils.read_image(ech_data["file_name"], format="BGR")
    transform_list = [
        T.Resize((512, 512)),
        # T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        # T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        # T.RandomBrightness(0.8, 1.2),
        # T.RandomContrast(0.8, 1.2),
        # T.RandomRotation([-10, 10]),
        # T.RandomSaturation(0.8, 1.2),
        # T.RandomLighting(scale=0.1),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    ech_data["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    aug_annotation = []
    for ech_ann in ech_data.pop("annotations"):
        aug_annotation.append(utils.transform_instance_annotations(ech_ann, transforms, image.shape[:2]))

    ech_data["instances"] = utils.annotations_to_instances(aug_annotation, image.shape[:2])
    return ech_data



class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=collate_fn)



# trainer = DefaultTrainer(cfg)


trainer = CustomTrainer(cfg)

# for name, param in trainer.model.named_parameters():
#     print(f"Layer: {name}, Requires Grad: {param.requires_grad}")
from detectron2.checkpoint import DetectionCheckpointer


if TRAIN_DETECTION:
    print("*************")
    print(cfg.MODEL.WEIGHTS)
    print("*************")
    model = trainer.build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    trainer.resume_or_load(resume=False)
    trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "../output/model_final.pth")
print("cfg.MODEL.WEIGHTS", cfg.MODEL.WEIGHTS)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
predictor = DefaultPredictor(cfg)

if EVAL_DETECTION:

    evaluator = COCOEvaluator("data_detection_train", tasks=cfg, distributed=False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "data_detection_train")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))














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
  def __init__(self, set_name, data_list, is_aug=False):

      self.tran = transforms.Compose([
          transforms.ToTensor(), # Converting the image to tensor and change the image format (Channels-Last => Channels-First)
      ])

      self.set_name = set_name
      self.data = data_list
      self.instance_map = []
      self.is_aug = is_aug
      if self.is_aug:
          print("Augmentation Enabled")
      else:
          print("Augmentation Disabled")
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
        if self.is_aug:
            if random.random() > 0.50:  # Augment only 35% of the training data
                if random.random() > 0.5:
                    img = cv2.flip(img, 1)
                    mask = cv2.flip(mask, 1)
                if random.random() > 0.5:
                    img = cv2.flip(img, 0)
                    mask = cv2.flip(mask, 0)
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

def get_plane_dataset(set_name='train', batch_size=4,is_aug=False):
    my_data_list = DatasetCatalog.get("data_detection_{}".format(set_name))
    dataset = PlaneDataset(set_name, my_data_list,is_aug)
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
# Set the hyperparameters
num_epochs = 30
batch_size = 2
learning_rate = 1e-3
weight_decay = 1e-5


model = MyModel() # initialize the model
model = model.cuda() # move the model to GPU


learning_rate = 1e-3
wt = torch.load('{}/output/final_segmentation_model.pth'.format(BASE_DIR))
# wt = torch.load('{}/output_v3/final_segmentation_model.pth'.format(BASE_DIR))
model.load_state_dict(wt)


loader, _ = get_plane_dataset('train', batch_size, False) # initialize data_loader
crit = nn.BCEWithLogitsLoss() # Define the loss function
optim = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # Initialize the optimizer as SGD


def get_lr(optim):
    cur_lr = 0
    for param_group in optim.param_groups:
        cur_lr = param_group['lr']
    return cur_lr

loss_ctr =0
best_loss = 100
# start the training procedure
if TRAIN_SEGMENTATION:




    for epoch in range(num_epochs):
      cur_lr = get_lr(optim)
      print("Epoch: {}, cur_lr: {}".format(epoch, cur_lr))
      total_loss = 0
      for (img, mask) in tqdm(loader):
        img = torch.tensor(img, device=torch.device('cuda'), requires_grad = True)
        mask = torch.tensor(mask, device=torch.device('cuda'), requires_grad = True)
        pred = model(img)
        loss = crit(pred, mask)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.cpu().data
      avg_loss = total_loss / len(loader)
      print("Epoch: {}, Loss: {}".format(epoch, avg_loss))
      if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), '{}/output/{}_{}_segmentation_model.pth'.format(BASE_DIR, epoch,np.round(best_loss, 4)))
      else:
          loss_ctr +=1
          if loss_ctr > 2:
              loss_ctr = 0
              for param_group in optim.param_groups:
                if cur_lr > 1e-6:
                    param_group['lr'] /= 2
                    cur_lr = param_group['lr']
    torch.save(model.state_dict(), '{}/output/final_segmentation_model.pth'.format(BASE_DIR))



# eval


batch_size = 8
#
model = get_seg_model('{}/output/final_segmentation_model.pth'.format(BASE_DIR))

model = model.eval()  # chaning the model to evaluation mode will fix the bachnorm layers
loader, dataset = get_plane_dataset('train', batch_size)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def iou(gt, pd):
    sigmoid(pd)
    mask1 = gt.astype(bool)
    mask2 = pd.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return union
    else:
        iou = intersection / union
        return iou

total_iou = 0

ctr = 0
global_iou = 0

if EVAL_SEGMENTATION:
    for (img, mask) in tqdm(loader):
        with torch.no_grad():
            img = img.cuda()
            mask = mask.cuda()
            mask = torch.unsqueeze(mask, 1).detach()
            pred = model(img).cpu().detach()
            for i in range(img.shape[0]):
                cur_pred = np.array(pred[i].cpu())[0]
                cur_mask = np.array(mask[i].cpu())[0].squeeze()

                cur_pred = np.where(cur_pred > 0.5, 255.0, 0.0)
                cur_mask = np.where(cur_mask > 0.5, 255.0, 0.0)
                ctr += 1
                global_iou += iou(cur_mask, cur_pred)

            '''
            ## Complete the code by obtaining the IoU for each img and print the final Mean IoU
    '''

    print("\n #images: {}, Mean IoU: {}".format(ctr, global_iou / ctr))









# PART3

'''
# Define a new function to obtain the prediction mask by passing a sample data
# For this part, you need to use all the previous parts (predictor, get_instance_sample, data preprocessings, etc)
# It is better to keep everything (as well as the output of this funcion) on gpu as tensors to speed up the operations.
# pred_mask is the instance segmentation result and should have different values for different planes.
# TODO: approx 35 lines
'''



# model = get_seg_model()



def get_segment_for_crop(idx, y1, x1, y2, x2, img, model, device):
    ech_crop = cv2.resize(img[x1:x2, y1:y2], (128, 128), interpolation=cv2.INTER_AREA)
    pre_ech_crop = torch.unsqueeze(torch.tensor(transforms.ToTensor()(ech_crop), device=torch.device(device)), 0)
    pd_crop = model(pre_ech_crop).detach().cpu().numpy()[0]

    pd_crop = cv2.resize(pd_crop[0], (y2 - y1, x2 - x1), interpolation=cv2.INTER_AREA)

    pd_crop = sigmoid(pd_crop)
    pd_crop = np.where(pd_crop > 0.5, 255.0, 0.0)
    pd_crop[pd_crop == 255.0] = idx + 1
    return pd_crop


def get_prediction_mask(data):
    if is_gpu_available():
        device = "cuda"
    else:
        device = "cpu"

    img = cv2.imread(data['file_name'])

    # Getting prediction masks

    height = data['height']
    width = data['width']

    pd_masks = np.zeros([height, width])
    gt_mask = np.zeros([height, width])
    img_crops = []

    if len(data['annotations']) > 0:
        for idx, ech_ann in enumerate(data['annotations']):
            y1, x1, y2, x2 = [int(i) for i in ech_ann['bbox']]
            pd_masks[x1:x2, y1:y2] = get_segment_for_crop(idx, y1, x1, y2, x2, img, model, device)

        # Getting GT masks
        for idx, ech_ann in enumerate(data['annotations']):
            y1, x1, y2, x2 = [int(i) for i in ech_ann['bbox']]
            local_gt = detectron2.utils.visualizer.GenericMask(ech_ann['segmentation'], height, width).mask

            gt_mask[x1:x1 + x2, y1:y1 + y2] = np.maximum(gt_mask[x1:x1 + x2, y1:y1 + y2],
                                                         local_gt[x1:x1 + x2, y1:y1 + y2] * (idx + 1))
            # gt_mask[x1:x1+x2, y1:y1+y2] = local_gt[x1:x1+x2, y1:y1+y2]*(idx+1)

    else:
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox_pred = predictor(img)['instances']
        # return bbox_pred

        for idx in range(len(bbox_pred)):
            y1, x1, y2, x2 = np.array([int(i) for i in list(bbox_pred[idx]._fields['pred_boxes'])[0].cpu().numpy()])

            pd_masks[x1:x2, y1:y2] = get_segment_for_crop(idx, y1, x1, y2, x2, img, model, device)

    gt_mask = torch.tensor(gt_mask, device=torch.device(device))
    pd_masks = torch.tensor(pd_masks, device=torch.device(device))

    gt_mask = normalize_image(gt_mask)
    pd_masks = normalize_image(pd_masks)

    return img, gt_mask, pd_masks



'''
# ref: https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
# https://www.kaggle.com/c/airbus-ship-detection/overview/evaluation
'''
def rle_encoding(x):
    '''
    x: pytorch tensor on gpu, 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = torch.where(torch.flatten(x.long())==1)[0]
    if(len(dots)==0):
      return []
    inds = torch.where(dots[1:]!=dots[:-1]+1)[0]+1
    inds = torch.cat((torch.tensor([0], device=torch.device('cuda'), dtype=torch.long), inds))
    tmpdots = dots[inds]
    inds = torch.cat((inds, torch.tensor([len(dots)], device=torch.device('cuda'))))
    inds = inds[1:] - inds[:-1]
    runs = torch.cat((tmpdots, inds)).reshape((2,-1))
    runs = torch.flatten(torch.transpose(runs, 0, 1)).cpu().data.numpy()
    return ' '.join([str(i) for i in runs])


'''
# You need to upload the csv file on kaggle
# The speed of your code in the previous parts highly affects the running time of this part
'''

preddic = {"ImageId": [], "EncodedPixels": []}

'''
# Writing the predictions of the training set
'''


'''
# Writing the predictions of the test set
'''
if GEN_CSV:

    DatasetCatalog.remove('data_detection_train')
    MetadataCatalog.remove('data_detection_train')
    #
    DatasetCatalog.remove('data_detection_validation')
    MetadataCatalog.remove('data_detection_validation')
    #
    #
    DatasetCatalog.remove('data_detection_test')
    MetadataCatalog.remove('data_detection_test')

    for sel in ["train", "test"]:
        if sel == "train":
            temp_train, temp_validation = get_detection_data(sel)
            DatasetCatalog.register("data_detection_train", lambda sel=sel: temp_train)
            MetadataCatalog.get("data_detection_train").set(thing_classes=["plane"])

            DatasetCatalog.register("data_detection_validation", lambda sel=sel: temp_validation)
            MetadataCatalog.get("data_detection_validation").set(thing_classes=["plane"])
        else:
            DatasetCatalog.register("data_detection_" + sel, lambda sel=sel: get_detection_data(sel))
            MetadataCatalog.get("data_detection_" + sel).set(thing_classes=["plane"])

    train_metadata = MetadataCatalog.get("data_detection_train")
    validation_metadata = MetadataCatalog.get("data_detection_validation")
    test_metadata = MetadataCatalog.get("data_detection_test")

    my_data_list = DatasetCatalog.get("data_detection_{}".format('train'))
    for i in tqdm(range(len(my_data_list)), position=0, leave=True):
      sample = my_data_list[i]
      sample['image_id'] = sample['file_name'].split("/")[-1][:-4]
      img, true_mask, pred_mask = get_prediction_mask(sample)
      inds = torch.unique(pred_mask)
      if(len(inds)==1):
        preddic['ImageId'].append(sample['image_id'])
        preddic['EncodedPixels'].append([])
      else:
        for index in inds:
          if(index == 0):
            continue
          tmp_mask = (pred_mask==index)
          encPix = rle_encoding(tmp_mask)
          preddic['ImageId'].append(sample['image_id'])
          preddic['EncodedPixels'].append(encPix)

    my_data_list = DatasetCatalog.get("data_detection_{}".format('test'))
    for i in tqdm(range(len(my_data_list)), position=0, leave=True):
      sample = my_data_list[i]
      sample['image_id'] = sample['file_name'].split("/")[-1][:-4]
      img, true_mask, pred_mask = get_prediction_mask(sample)
      inds = torch.unique(pred_mask)
      if(len(inds)==1):
        preddic['ImageId'].append(sample['image_id'])
        preddic['EncodedPixels'].append([])
      else:
        for j, index in enumerate(inds):
          if(index == 0):
            continue
          tmp_mask = (pred_mask==index).double()
          encPix = rle_encoding(tmp_mask)
          preddic['ImageId'].append(sample['image_id'])
          preddic['EncodedPixels'].append(encPix)

    pred_file = open("{}/pred.csv".format(BASE_DIR), 'w')
    pd.DataFrame(preddic).to_csv(pred_file, index=False)
    pred_file.close()



cfg = get_cfg()
cfg.OUTPUT_DIR = "{}/output/".format(BASE_DIR)
cfg.SOLVER.MAX_ITER = 5000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.DATALOADER.NUM_WORKERS = 4

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.DATASETS.TRAIN = ("data_detection_train",)
cfg.DATASETS.TEST = ("data_detection_test",)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)

if RCNN_TRAIN:
    trainer.resume_or_load(resume=False)   #default False, ensures loading from cfg.MODEL.WEIGHTS
    trainer.train()


if RCNN_EVAL:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "mrcnn_model_final.pth")
    print("cfg.MODEL.WEIGHTS", cfg.MODEL.WEIGHTS)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("data_detection_train", tasks=cfg, distributed=False, output_dir= cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "data_detection_train")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))









