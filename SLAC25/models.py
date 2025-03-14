import torch
import torch.nn as nn
import torchvision.models as models

######################### Simple CNN Model ###############################

class BaselineCNN(nn.Module):
    def __init__(self, num_classes, keep_prob):
        super(BaselineCNN, self).__init__()
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        # L1 ImgIn shape=(?, input_size, input_size, 3)
        #    Conv     -> (?, input_size, input_size, 32)
        #    Pool     -> (?, input_size//2, input_size//2, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - keep_prob))
        # L2 ImgIn shape=(?, input_size//2, input_size//2, 32)
        #    Conv      ->(?, input_size//2, input_size//2, 64)
        #    Pool      ->(?, input_size//4, input_size//4, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - keep_prob))
        # L3 ImgIn shape=(?, input_size//4, input_size//4, 64)
        #    Conv      ->(?, input_size//4, input_size//4, 128)
        #    Pool      ->(?, input_size//8, input_size//8, 128)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - keep_prob))
    
        # L4 Fully Connected Layer 128*input_size//8*input_size//8 inputs -> 256 outputs
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 64 * 64, 256, bias=True),  # Adjusted input size
            nn.ReLU(),
            nn.Dropout(p=1 - keep_prob)
        )
        # L5 Fully Connected Layer 512 inputs -> num_classes outputs
        self.fc2 = nn.Sequential(
            nn.Linear(256, num_classes, bias=True) # no need for softmax since the loss function is cross entropy
        )

    def forward(self, x):
        out = self.layer1(x) # Conv + batch norm -> ReLU -> MaxPool -> Dropout
        out = self.layer2(out) # Conv + batch norm -> ReLU -> MaxPool -> Dropout
        out = self.layer3(out) # Conv + batch norm -> ReLU -> MaxPool -> Dropout
        out = out.view(out.size(0), -1) # Flatten them for FC, should be
        out = self.fc1(out) # FC -> ReLU -> Dropout
        out = self.fc2(out) # FC -> logits for our criterion
        return out

    def summary(self):
        print(self)

######################### ResNet Model ###############################

class ResNet(nn.Module):
    def __init__(self, num_classes, keep_prob):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.fc_layer1 = nn.Sequential(
            nn.Linear(1000, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=1 - keep_prob),
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(256, num_classes)
        )

    def transfer_learn(self):
        '''Function to transfer learn the model'''
        print('Transfer learning..., freezing all parameters of resnet50')
        # freeze all parameters of resnet50
        for param in self.resnet50.parameters():
            param.requires_grad = False

        print('Unfreezing layer 4 and fc layer of resnet50')
        # unfreeze layer 4 and fc layer of resnet50
        for param in self.resnet50.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet50.fc.parameters():
            param.requires_grad = True

        print('Unfreezing fc_layer1 and fc_layer2')
        # unfreeze fc_layer1 and fc_layer2
        for param in self.fc_layer1.parameters():
            param.requires_grad = True
        for param in self.fc_layer2.parameters():
            param.requires_grad = True
        print('Transfer learning complete')
        
    def forward(self, x):
        x = self.resnet50(x)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        return x
    
    def summary(self):
        print(self)