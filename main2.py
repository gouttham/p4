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
OUTPUT_DIR = '{}/output'.format(BASE_DIR)


os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_detection_data(set_name):
    data_dirs = '{}/data'.format(BASE_DIR)
    train_dirs = '{}/data/train'.format(BASE_DIR)
    test_dirs = '{}/data/test'.format(BASE_DIR)
    dataset = []
    json_file = os.path.join(data_dirs, "train.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    if set_name == "test":
          # Handle test set where annotations do not exist
          for img_file in os.listdir(test_dirs):
              if img_file.endswith('.png'):
                  img_path = os.path.join(test_dirs, img_file)
                  with Image.open(img_path) as img:
                      width, height = img.size
                  dataset.append({
                      "file_name": img_path,
                      "image_id": os.path.splitext(img_file)[0],
                      "height": height,
                      "width": width,
                      "annotations": []  # Test data has no annotations
                  })

    else:
        images = {}
        for ann in imgs_anns:
            image_id = ann["image_id"]
            if image_id not in images:
                img_path = os.path.join(train_dirs, ann["file_name"])
                with Image.open(img_path) as img:
                    width, height = img.size
                images[image_id] = {
                    "file_name": img_path,
                    "image_id": image_id,
                    "height": height,
                    "width": width,
                    "annotations": []
                }
            # Append annotation to the corresponding image entry
            images[image_id]["annotations"].append({
                "bbox": ann["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 0,
                "segmentation": ann.get("segmentation", None),
            })

        dataset = list(images.values())

    return dataset

for d in ["train", "val", "test"]:
    if "plane_" + d in DatasetCatalog.list():
        DatasetCatalog.remove("plane_" + d)
    DatasetCatalog.register("plane_" + d, lambda d=d: get_detection_data(d))
    MetadataCatalog.get("plane_" + d).set(thing_classes=["plane"])
planes_metadata = MetadataCatalog.get("plane_train")






'''
# Set the configs for the detection part in here.
# TODO: approx 15 lines
'''
cfg = get_cfg()
cfg.OUTPUT_DIR = "{}/output/".format(BASE_DIR)

# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))  # baseline model
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("plane_train",)
cfg.DATASETS.TEST = ("plane_test",)

cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # baseline model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 5000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1


from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data import build_detection_train_loader
import copy

# Custom data augmentation
# reference: https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html

def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [
                      T.Resize((512,512)),
                      T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                      T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                      T.RandomRotation([-20, 20]),
                      T.RandomBrightness(0.7, 1.3),
                      T.RandomContrast(0.7, 1.3),
                      T.RandomSaturation(0.7, 1.3),
                      T.RandomLighting(0.7)
                     ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    annos = [
          utils.transform_instance_annotations(ann, transforms, image.shape[:2])
          for ann in dataset_dict.pop("annotations")
          if ann.get("iscrowd", 0) == 0
          ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=mapper)



'''
# Create a DefaultTrainer using the above config and train the model
# TODO: approx 5 lines
'''

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
trainer = CustomTrainer(cfg)


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
predictor = DefaultPredictor(cfg)




# Caching Train Data
train_crop = []
train_dicts = get_detection_data("train")

for v in train_dicts:
    record = {}
    record["file_name"] = v["file_name"]
    objs = []

    # Load image
    try:
        img = Image.open(v["file_name"]).convert('RGB')
    except IOError:
        print(f"Warning: Image {v['file_name']} could not be opened.")
        continue

    for ann in v["annotations"]:
        img_seg = ann["segmentation"]
        img_bbox = ann["bbox"]  # [x, y, width, height]
        x, y, w, h = map(int, img_bbox)
        x2, y2 = x + w, y + h

        # Adjust segmentation coordinates relative to the cropped image
        new_seg = []
        for seg in img_seg:
            adjusted_seg = []
            for j in range(0, len(seg), 2):
                adjusted_x = seg[j] - x
                adjusted_y = seg[j+1] - y
                # Ensure coordinates are within the cropped image
                adjusted_x = max(0, min(adjusted_x, w))
                adjusted_y = max(0, min(adjusted_y, h))
                adjusted_seg.append(adjusted_x)
                adjusted_seg.append(adjusted_y)
            new_seg.append(adjusted_seg)

        # Crop and resize the image
        cropped_img = img.crop((x, y, x2, y2))
        obj_img = cropped_img.resize((128, 128), Image.LANCZOS)
        obj_img = np.array(obj_img)

        # Generate the mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for adjusted_seg in new_seg:
            if len(adjusted_seg) < 6:
                # A polygon must have at least 3 points (6 values)
                continue
            poly = np.array(adjusted_seg).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [poly], 1)

        # Resize the mask to (128, 128)
        obj_mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)

        # Ensure the image has 3 channels
        if obj_img.ndim == 2:
            obj_img = np.stack([obj_img]*3, axis=-1)

        obj = {
            "obj_img": obj_img,
            "obj_mask": obj_mask,
        }
        objs.append(obj)

    record["annotations"] = objs
    train_crop.append(record)


# Caching Test Data
test_crop = []
test_dicts = get_detection_data("test")

for pic in test_dicts:
    record = {}
    record["file_name"] = pic["file_name"]
    objs = []

    # Load the image once using OpenCV (BGR)
    img_bgr = cv2.imread(pic["file_name"])
    if img_bgr is None:
        print(f"Warning: Image {pic['file_name']} could not be loaded.")
        continue

    # Convert BGR to RGB for PIL Image compatibility
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    outputs = predictor(img_bgr)

    instances = outputs["instances"]
    num_instances = len(instances)

    for i in range(num_instances):
        # Get the bounding box coordinates
        bbox = instances.pred_boxes[i].tensor.cpu().numpy().flatten()
        x1, y1, x2, y2 = bbox.astype(int)
        width = x2 - x1
        height = y2 - y1

        # Validate bounding box coordinates
        if width <= 0 or height <= 0:
            print(f"Invalid bbox with width {width} and height {height} in image {pic['file_name']}. Skipping.")
            continue

        cropped_img = pil_img.crop((x1, y1, x2, y2))
        obj_img = cropped_img.resize((128, 128), Image.LANCZOS)
        obj_img = np.array(obj_img)
        # Ensure the image has 3 channels
        if obj_img.ndim == 2:
            obj_img = np.stack([obj_img]*3, axis=-1)

        # Initialize the mask to zeros
        obj_mask = np.zeros((128, 128), dtype=np.uint8)

        obj = {
            "bbox": [x1, y1, x2, y2],
            "obj_img": obj_img,
            "obj_mask": obj_mask,
        }
        objs.append(obj)

    record["annotations"] = objs
    test_crop.append(record)



'''
# Write a function that returns the cropped image and corresponding mask regarding the target bounding box
# idx is the index of the target bbox in the data
# high-resolution image could be passed or could be load from data['file_name']
# You can use the mask attribute of detectron2.utils.visualizer.GenericMask
#     to convert the segmentation annotations to binary masks
# TODO: approx 10 lines
'''

def get_instance_sample(data, idx, prepared_imageset):
  for i in prepared_imageset:
    if i["file_name"] == data["file_name"]:
      obj_img = i["annotations"][idx]["obj_img"]
      obj_mask = i["annotations"][idx]["obj_mask"]
      break
  return obj_img, obj_mask

'''
# We have provided a template data loader for your segmentation training
# You need to complete the __getitem__() function before running the code
# You may also need to add data augmentation or normalization in here
'''

class PlaneDataset(Dataset):
    def __init__(self, set_name, data_list,prepared_imageset):
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor()  # Converting the image to tensor and change the image format (Channels-Last => Channels-First)
            ]
        )
        self.set_name = set_name
        self.data = data_list
        self.predata = prepared_imageset
        self.instance_map = []
        for i, d in enumerate(self.data):
            for j in range(len(d["annotations"])):
                self.instance_map.append([i, j])

    """
  # you can change the value of length to a small number like 10 for debugging of your training procedure and overfeating
  # make sure to use the correct length for the final training
  """

    def __len__(self):
        return len(self.instance_map)

    def numpy_to_tensor(self, img, mask):
        if self.transforms is not None:
            img = self.transforms(img)
        img = torch.tensor(img, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.float)
        return img, mask

    """
  # Complete this part by using get_instance_sample function
  # make sure to resize the img and mask to a fixed size (for example 128*128)
  # you can use "interpolate" function of pytorch or "numpy.resize"
  # TODO: 5 lines
  """

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self.instance_map[idx]
        data = self.data[idx[0]]
        ann_index = idx[1]
        img, mask = get_instance_sample(data, ann_index,self.predata)

        img, mask = self.numpy_to_tensor(img, mask)
        img = img.reshape((3,128,128))
        if len(mask.shape)==1:
            import pdb
            pdb.set_trace()
        mask = mask.reshape((1,128,128))

        return img, mask

def get_plane_dataset(set_name, prepared_imageset,batch_size=2):
    my_data_list = DatasetCatalog.get("plane_{}".format(set_name))
    dataset = PlaneDataset(set_name, my_data_list,prepared_imageset)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0,
                                              pin_memory=True, shuffle=True)
    return loader, dataset

def get_prediction_dataset(set_name, data_list,prepared_imageset,batch_size=2):
    dataset = PlaneDataset(set_name, data_list,prepared_imageset)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0,
                                              pin_memory=True, shuffle=False)
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
model = MyModel().cuda()
model.load_state_dict(torch.load('{}/output_v3/final_segmentation_model.pth'.format(BASE_DIR)))
model = model.eval() # changing the model to evaluation mode will fix the batchnorm layers
loader, dataset = get_plane_dataset('train', train_crop, batch_size)


"""
# Define a new function to obtain the prediction mask by passing a sample data
# For this part, you need to use all the previous parts (predictor, get_instance_sample, data preprocessings, etc)
# It is better to keep everything (as well as the output of this funcion) on gpu as tensors to speed up the operations.
# pred_mask is the instance segmentation result and should have different values for different planes.
# TODO: approx 35 lines
"""

def get_prediction_mask(data, prepared_imageset):
    # Prepare sample
    sample = [name for name in prepared_imageset if name["file_name"] == data["file_name"]] if not data["annotations"] else [data]

    # Ground truth mask
    height, width = data["height"], data["width"]
    gt_mask = (
        torch.from_numpy(
            detectron2.utils.visualizer.GenericMask(
                [ann["segmentation"][0] for ann in data["annotations"]],
                height,
                width,
            ).mask
        )
        if data["annotations"]
        else None
    )

    # Load model
    model = MyModel().cuda()
    model.load_state_dict(torch.load(f"{BASE_DIR}/output/final_segmentation_model.pth"))
    model.eval()

    # Predictions
    loader, _ = get_prediction_dataset("prediction", sample, prepared_imageset, batch_size=8)
    pred_data = []
    for img, _ in loader:
        with torch.no_grad():
            preds = nn.Sigmoid()(model(img.cuda()))
            pred_data.extend((pred.squeeze().cpu().numpy() >= 0.4).astype(int) for pred in preds)

    # Combine predictions with ground truth
    pred_mask, gt = combine_predictions_with_gt(data, pred_data, height, width)
    gt_tensor = torch.from_numpy(gt).cuda()
    return Image.open(data["file_name"]), gt_mask.cuda() if gt_mask is not None else gt_tensor, gt_tensor

def combine_predictions_with_gt(data, pred_data, height, width):
    gt = np.zeros((height, width), dtype=np.int32)
    for idx, pred in enumerate(pred_data):
        if idx >= len(data["annotations"]):  # Skip extra predictions
            break
        x, y, w, h = map(int, data["annotations"][idx]["bbox"])
        resized_pred = cv2.resize(pred, (w, h))
        gt[y : y + h, x : x + w] = np.where(resized_pred > 0.5, idx + 1, gt[y : y + h, x : x + w])
    return pred_data, gt

def get_prediction_test(data, prepared_imageset):
    img, gt_mask, pred_mask = get_prediction_mask(data, prepared_imageset)
    for name in prepared_imageset:
        if name["file_name"] == data["file_name"]:
            num_annotations = len(name["annotations"])
            num_preds = len(pred_mask)
            # Adjust annotation length if needed
            if num_preds > num_annotations:
                name["annotations"].extend([{"obj_mask": None} for _ in range(num_preds - num_annotations)])
            # Assign predictions
            for idx, pred in enumerate(pred_mask):
                name["annotations"][idx]["obj_mask"] = pred
    return img, gt_mask, pred_mask


'''
# Visualise the output prediction as well as the GT Mask and Input image for a sample input
# TODO: approx 10 lines
'''

idx = 2
my_data_list = DatasetCatalog.get("plane_{}".format('test'))
sample = my_data_list[idx]
img, gt_mask, pred_mask = get_prediction_test(sample, test_crop)



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
my_data_list = DatasetCatalog.get("plane_{}".format('train'))
for i in tqdm(range(len(my_data_list)), position=0, leave=True):
  sample = my_data_list[i]
  sample['image_id'] = sample['file_name'].split("/")[-1][:-4]
  img, true_mask, pred_mask = get_prediction_mask(sample, train_crop)
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
'''
'''
# Writing the predictions of the test set
'''

my_data_list = DatasetCatalog.get("plane_{}".format('test'))
for i in tqdm(range(len(my_data_list)), position=0, leave=True):
  sample = my_data_list[i]
  sample['image_id'] = sample['file_name'].split("/")[-1][:-4]

  img, true_mask, pred_mask = get_prediction_test(sample,test_crop)
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