import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

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
    def __init__(self, num_classes, keep_prob, hidden_num = 256, input_size = 224):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.hidden_num = hidden_num
        self.keep_prob = keep_prob
        self.input_size = input_size

        # initialize with pretrained weights
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.fc_layer1 = nn.Sequential(
            nn.Linear(1000, self.hidden_num),
            nn.BatchNorm1d(self.hidden_num),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob),
        )
        
        self.fc_layer2 = nn.Sequential(
            nn.Linear(self.hidden_num, self.hidden_num // 2),
            nn.BatchNorm1d(self.hidden_num // 2),
            nn.ReLU(),
            nn.Dropout(p=1 - self.keep_prob),
        )

        self.fc_layer3 = nn.Sequential(
            nn.Linear(self.hidden_num // 2, num_classes),
        )
        
    def forward(self, x):
        x = self.resnet(x) # input_size x input_size x 3 -> 1000
        x = self.fc_layer1(x) # 1000 -> hidden_num
        x = self.fc_layer2(x) # hidden_num -> hidden_num // 2
        x = self.fc_layer3(x) # hidden_num // 2 -> num_classes
        return x
    def transfer_learn_phase1(self):
        '''
        Initiali training phase with 224x224 input size and unfreeze layers 2-4 and Fully connected layers
        '''
        print('Phase 1 transfer learning setup')

        # freeze params
        for param in self.resnet.parameters():
            param.requires_grad = False

        # unfreeze layers 2-4
        layers_to_unfreeze = [self.resnet.layer4, 
                              self.resnet.layer3, 
                              self.resnet.layer2,
                              ]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        
        # unfreeze fully connected layers
        for layer in [self.fc_layer1, self.fc_layer2, self.fc_layer3]:
            for param in layer.parameters():
                param.requires_grad = True

        print('Phase 1 setup complete: Unfroze ResNet layers 2-4 and FC layers')

    def transfer_learn_phase2(self):
        '''Second training phase with 512x512 images
        Freezes layer 2 and keeps layers 3-4 unfrozen'''
        print('Phase 2 Transfer Learning Setup...')
        # First freeze everything
        for param in self.resnet.parameters():
            param.requires_grad = False

        # freeze layer 2 and 3
        for layer in [self.resnet.layer2, self.resnet.layer3]:
            for param in layer.parameters():
                param.requires_grad = False
        
        # unfreeze layer 4
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
            
        # unfreeze fully connected layers
        for layer in [self.fc_layer1, self.fc_layer2, self.fc_layer3]:
            for param in layer.parameters():
                param.requires_grad = True

        print('Phase 2 setup complete: Unfroze ResNet layers 3-4 and FC layers')
    
            
    def print_trainable_parameters(self):
        '''Function to print the trainable parameters'''
        for name, param in self.named_parameters():
            print(f'{name}: {"trainable" if param.requires_grad else "frozen"}')

    def print_model_summary(self):
        '''Print model summary only for the resnet part'''
        summary(self.resnet, (3, 224, 224))
