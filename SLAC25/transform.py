import random
import torch

from torchvision.transforms import v2
# from PIL import Image

tfConfig = {
    "rotation": {
        "on": True,
        "params": {
            "degrees": (-45, 45)
        }
    },
    "horizontal_flip": {
        "on": True
    },
    "vertical_flip": {
        "on": True
    },
    "gaussian_blur": {
        "on": True,
        "params": {
            "kernel_size": 5,
            "sigma": (0.1, 2.0)
        }
    }
}

class TransformV1():
  def __init__(self, image_size=512, transform=None, config=None, recordTransform=False):
    self.tf_log = None
    self.tf_config = None
    self.image_size = image_size
    self._setTfConfig(config)
    self._setTfLog(recordTransform) # use to record transformation info
    self.preprocessing = v2.Compose([
      v2.Resize((self.image_size, self.image_size), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
      v2.PILToTensor(),
      v2.ConvertImageDtype(torch.float32)
    ])

  def _setTfLog(self, recordTf):
    if not recordTf:
      return

    else:
      self.tf_log = {}
  
  def _setTfConfig(self, config):
    if not config:
      self.tf_config = tfConfig
    
    else:
      self.tf_config = config
  
  def _checkConfig(self):
    if not self.tf_config:
      return False
    
    else:
      return True
  
  def _recordTf(self, idx, action):
    if self.tf_config is None:
      return
    
    if not self.tf_config.get(idx):
       self.tf_config[idx] = []
    
    self.tf_config[idx].append(str(action))

  def _emptyRecord(self, idx):
    if self.tf_log is None:
      return
    
    if not self.tf_config.get(idx):
      return
    
    else:
      self.tf_config[idx].clear()

  def _getLog(self, idx):
    if self.tf_log is None:
      return
    
    if not self.tf_config.get(idx):
      return
    
    res = '\n'.join(map(str, self.tf_config[idx]))
    
    return res

  def _random_gaussian_blur(self, img, idx):
    """Gaussian blur with 50% probability"""
    if self._checkConfig() and self.tf_config["gaussian_blur"]["on"]:
      params = self.tf_config["gaussian_blur"]["params"]
      if random.random() < 0.5:
        self._recordTf(idx, f"Gaussian Blur: {str(params)}")
        return v2.GaussianBlur(**params)(img)

    return img

  def _random_rotation(self, img, idx):
    """Random rotation between -45 to 45 degrees with a 50% probability"""
    if self._checkConfig() and self.tf_config["rotation"]["on"]:
      params = self.tf_config["rotation"]["params"]
      if random.random() < 0.5:
        angle = random.uniform(*params["degrees"])
        self._recordTf(idx, f"Rotating by {angle:.2f} degrees")
        return v2.RandomRotation(degrees=(angle, angle))(img)
    return img

  def _random_horizontal_flip(self, img, idx):
    """Horizontal flip with 50% probability"""
    if self._checkConfig() and self.tf_config["horizontal_flip"]["on"]:
      if random.random() < 0.5:
        self._recordTf(idx, "Applying Horizontal Flip")
        return v2.RandomHorizontalFlip()(img)
      
    return img

  def _random_vertical_flip(self, img, idx):
    """Vertical flip with 50% probability"""
    if self._checkConfig() and self.tf_config["horizontal_flip"]["on"]:
      if random.random() < 0.5:
        self._recordTf(idx, "Applying Vertical Flip")
        return v2.RandomVerticalFlip()(img)
      
    return img
    