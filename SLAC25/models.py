import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
from torchvision.models.segmentation import fcn_resnet50

######################### ResNet Models ###############################
class ResNet(nn.Module):
    def __init__(self, num_classes, keep_prob, hidden_num=256):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_num = hidden_num
        self.keep_prob = keep_prob

        # 1) Load pretrained ResNet-50 and drop its head
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # → now outputs a 2048-d feature vector

        # 2) New fully connected head
        self.fc_layer1 = nn.Sequential(
            nn.Linear(2048, hidden_num),
            nn.BatchNorm1d(hidden_num),
            nn.ReLU(inplace=True),
            nn.Dropout(1 - keep_prob),
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(hidden_num, hidden_num // 2),
            nn.BatchNorm1d(hidden_num // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(1 - keep_prob),
        )
        self.fc_layer3 = nn.Linear(hidden_num // 2, num_classes)

    def forward(self, x):
        # x: (B, 3, H, W) → ResNet backbone → (B, 2048)
        x = self.resnet(x)
        x = self.fc_layer1(x)     # → (B, hidden_num) say 1024
        x = self.fc_layer2(x)     # → (B, hidden_num//2) say 512
        x = self.fc_layer3(x)     # → (B, num_classes)
        return x
    
    # only one transfer learning phase for the last layer
    def transfer_learn_phase(self):
        """
        Freeze all Conv layers except layer3–4; train those plus the new FC head.
        """
        # Freeze entire backbone
        for p in self.resnet.parameters():
            p.requires_grad = False

        # Unfreeze layers 4
        for layer in self.resnet.layer4:
            for p in layer.parameters():
                p.requires_grad = True

        # Unfreeze our head
        for module in (self.fc_layer1, self.fc_layer2, self.fc_layer3):
            for p in module.parameters():
                p.requires_grad = True

        print(f"{'#'*3} Transfer learning phase completed {'#'*3}")


# create an VAE model using resnet50 encoder and UNet decoder

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 8 * 8),
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 64x64 -> 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), # 128x128 -> 256x256
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1), # 256x256 -> 512x512
            nn.Sigmoid()  # put pixels in the range [0,1]
        )

    def encode(self, x):
        x = self.resnet(x)
        h = self.fc1(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return z, recon, mu, logvar


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