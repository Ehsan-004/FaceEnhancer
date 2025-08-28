import torch
import torch.nn as nn
from piq import ssim


class FaceUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # feature size / 2
        # channel 3 -> 64
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # /2
        )
        
        # feature size * 2
        # channel 64 -> 3
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # after concatenating output of up 3 with the input image
        self.conv1 = nn.Conv2d(6, 3, kernel_size=3, padding=1, stride=1)  # 3 channels
        
        # ===============================================================================================
        
        # feature size / 2
        # channel 64 -> 128
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),  # [128]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # /2
        )
        
        # feature size * 2
        # channel 128 -> 64
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # after concatenating output of up 2 with output of down 1
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1)  # 64 channels
        
        # ===============================================================================================

        # feature size / 2
        # channel 128 -> 256
        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),  # [256]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # /2
        )
        
        # feature size * 2
        # channel 256 -> 128
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # *2 , [64]
            nn.ReLU(),
        )
        
        # after concatenating output of up 1 with output of down 2
        self.conv3 = nn.Conv2d(256, 128,  kernel_size=3, padding=1, stride=1)  # 128 channels
        
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, x):
        identity0 = x  # c = 3
        
        # ==--==--==--==-
        x = self.down1(x)  # c = 64
        identity1 = x  # c = 64
        
        # ==--==--==--==-
        x = self.down2(x)  # c = 128
        identity2 = x  # c = 128
        
        # ==--==--==--==-
        x = self.down3(x)  # c = 256
        
        # ==--==--==--==-
        x = self.up3(x)  # c = 128
        x = torch.cat([x, identity2], dim=1)  # c = 256
        x = self.conv3(x)  # c = 128  |  recover the channels again
        
        # ==--==--==--==-
        x = self.up2(x)  # c = 64
        x = torch.cat([x, identity1], dim=1)  # c = 128
        x = self.conv2(x)  # c = 64  |  recover the channels again
        
        # ==--==--==--==-
        x = self.up1(x)
        x = torch.cat([x, identity0], dim=1)  # c = 6
        x = self.sigmoid(self.conv1(x))  # c = 3  |  recover the channels again
        
        return x


class FaceUNetLoss(nn.Module):
    def __init__(self, lambda_ssim: float = 0.3):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        mse_part = self.mse_loss(pred, target)
        ssim_part = 1 - ssim(pred, target)
        return mse_part + self.lambda_ssim * ssim_part
