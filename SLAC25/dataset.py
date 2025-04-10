import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import pytz
import json
# import random
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from datetime import datetime

from SLAC25.transform import TransformV1


class ImageDataset(Dataset):
  def __init__(self, csvfilePath, image_size=512, transform=None, config=None, recordTransform=False):
    self.csvfilePath = csvfilePath
    self.dataframe = pd.read_csv(csvfilePath)
    self.datasetType = self._checkTrainTest()
    self.config = self._loadConfig()
    self.image_size = image_size

    if transform is None:
      self.transform = TransformV1(
        image_size=self.image_size, # pass the image size
        config=self.config.get("transform", None), # pass the config
        recordTransform=recordTransform # pass the recordTransform
      )
    else:
      self.transform = transform
    
    self.datasize = self.dataframe.shape[0]
    self.numLabel = self.dataframe['label_id'].nunique()
    self.labeldict = {idnum: self.dataframe.index[self.dataframe['label_id'] == idnum].to_list() for idnum in self.dataframe['label_id'].value_counts().index}

    dataLastModified = os.stat(self.csvfilePath).st_mtime
    self.dataLastModified = datetime.fromtimestamp(dataLastModified).strftime('%Y-%m-%d %H:%M:%S')
  
  def __len__(self):
    return self.datasize
  
  def __getitem__(self, idx):
    self.transform._emptyRecord(idx) # clean up the transform record of past imgs

    img = Image.open(self.getImagePath(idx))
    img = self.transform.preprocessing(img) # apply the basic transform to the image

    # apply augmentations
    img = self.transform._random_rotation(img, idx)
    img = self.transform._random_horizontal_flip(img, idx)
    img = self.transform._random_vertical_flip(img, idx)
    img = self.transform._random_gaussian_blur(img, idx)

    label = torch.tensor(self.getLabelId(idx), dtype = torch.long) # convert the label to a tensor
    return img, label
  
  def getImagePath(self, idx):
    row = self.dataframe.iloc[idx]
    return row['image_path']
  
  def getLabelId(self, idx):
    row = self.dataframe.iloc[idx]
    return row['label_id']
  
  def _checkTrainTest(self):
    if "test_info" in self.csvfilePath:
      return "TestSet"
    
    if "train_info" in self.csvfilePath:
      return "TrainSet"
    
    else:
      return "Others"
  
  def _loadConfig(self):
    package_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(package_root, "..", "config.json")
    config_path = os.path.abspath(config_path)

    with open(config_path, "r") as f:
      allConfig = json.load(f)
    
    config = allConfig["dataset"]["ImageDataset"]

    return config
  
  def _setupTransform(self, transform, config, recordTransform):
    if not transform:
      self.transform = TransformV1(config, recordTransform)
    
    else:
      self.transform = transform

  
  def visualizeAndSave(self, idx, savedPath=None):
    if savedPath is None:
      package_root = os.path.dirname(os.path.abspath(__file__))
      savedPath = os.path.join(package_root, "..", "img")
      savedPath = os.path.abspath(savedPath)

    os.makedirs(savedPath, exist_ok=True) # make a dir if not exists
    titleBefT_params = self.config["visualizeAndSave"]["title_before_transform"]
    titleAftT_params = self.config["visualizeAndSave"]["title_after_transform"]
    label_params = self.config["visualizeAndSave"]["label_text_params"]
    label_params['s'] = label_params['s'].format(self.getLabelId(idx))
    label_transformlogs = self.config["visualizeAndSave"]["label_transform_logs"]

    
    timeNow = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d%H%M%S")
    filename = os.path.join(savedPath ,f"{self.datasetType}_{timeNow}_{idx}.png")
    filename = os.path.abspath(filename)

    ### Image Before Transform ###
    imgBefT = Image.open(self.getImagePath(idx))
    
    ### Image After Transform ###
    res = self.__getitem__(idx)
    imgAftT = res[0].clone().detach().cpu()
    label_transformlogs['s'] = label_transformlogs['s'].format(self.transform._getLog(idx))
    imgAftTnp = imgAftT.numpy().transpose(1, 2, 0)


    ### Plot ###
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(imgBefT)
    ax[0].axis('off')
    ax[1].imshow(imgAftTnp)
    ax[1].axis('off')
    fig.text(**titleBefT_params)
    fig.text(**titleAftT_params)
    fig.text(**label_params)
    fig.text(**label_transformlogs)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

  
  def summary(self):
    '''
    prints a summary of the dataset
    '''
    print(f"\n{'='*40}")
    print(f"{' '*12}Dataset Summary")
    print(f"{'-'*40}")
    print(f"File Path:    {self.csvfilePath}")
    print(f"Last Updated: {self.dataLastModified}")
    print(f"{'-'*40}")
    print(f"Sample Sizes: {self.datasize}")
    print(f"Label Types: {self.numLabel}")
    for l in range(self.numLabel):
      print(f"Label {l}: {len(self.labeldict[l])} | {len(self.labeldict[l]) / self.datasize * 100:.2f}%")
    print(f"{'='*40}")
 


if __name__ == "__main__":
  package_root = os.path.dirname(os.path.abspath(__file__))
  data_path = os.path.join(package_root, "..", "data", "train_info.csv")
  data_path = os.path.abspath(data_path)
  testData = ImageDataset(data_path, transform=None, config=None, recordTransform=True)

  testData.visualizeAndSave(123)
  # print(data_path)
