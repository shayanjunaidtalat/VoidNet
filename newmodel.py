import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
#!Change of in_channels from 3 to 1
class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #!Added by Muhammad
        self.downs.append(nn.Sequential(nn.Conv2d(3, 64, 7, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),))
        in_channels = 64
        # Down part of UNET
        #!Added by Muhammad ie He is doing the first downsampling manually
        for feature in features[1:]:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2,))
            self.ups.append(DoubleConv(feature*2, feature))
        #!Added by Muhammad
        self.ups.append(
                nn.ConvTranspose2d(
                    64, 64, 8, 2, 1
                )
            )
        #self.ups.append(DoubleConv(64, 64))
        #!Added by Muhammad

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        #!NEW
        # downscale = nn.Conv2d(3,64,7,stride = 2,padding=0).to(device='cuda')
        # x = downscale(x)
        #!NEW

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups)-2, 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        #!NEW
        x = self.ups[-1](x)
        #comment this layer
        #x = self.ups[-1](x)
        #comment this layer        
        x = self.final_conv(x)#!         ORIGINAL RETURN LINE
#!        return self.final_conv(x)
        # x = upscale(x)
        #!NEW
#!         ORIGINAL RETURN LINE
#!        return self.final_conv(x)
        return x

def test():
    # x = torch.randn((3, 1, 161, 161))
    # #!Change of in_channels from 3 to 1
    # model = UNET(in_channels=64, out_channels=1)
    # preds = model(x)
    # assert preds.shape == x.shape
    a=1

if __name__ == "__main__":
    test()