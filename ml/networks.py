"""
What-is-it : CartoonGAN
Why-of-it  : A minimal implementation
Author     : Nirmalendu B Patra
"""
import torch.nn as nn
import torch.nn.functional as TNF

class ConvDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDown, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        self.conv_2 = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return TNF.relu(self.norm(self.conv_2(self.conv_1(x))))

class ConvUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUp, self).__init__()
        self.conv_1 = nn.ConvTranspose2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         output_padding=1)
        self.conv_2 = nn.ConvTranspose2d(in_channels=out_channels,
                                         out_channels=out_channels,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         output_padding=1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return TNF.relu(self.norm(self.conv_2(self.conv_1(x))))

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=256,
                                out_channels=256,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.norm_1 = nn.BatchNorm2d(256)
        self.conv_2 = nn.Conv2d(in_channels=256,
                                out_channels=256,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.norm_1 = nn.BatchNorm2d(256)

    def forward(self, x):
        return self.norm_2(self.conv_2(TNF.relu(self.norm_1(self.conv_1(x))))) + x

class ConvFlat(nn.Module):
    def __init__(self):
        super(ConvFlat, self).__init__()
        # k7n64s1 from 3 color input image
        self.conv = nn.Conv2d(in_channels=3,
                              out_channels=64,
                              kernel_size=7,
                              stride=1,
                              padding=(7 - 1) // 2)
        self.norm = nn.BatchNorm2d(64)

    def forward(self, x):
        return TNF.relu(self.norm(self.conv(x)))

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # k7n64s1 from 3 color input image
        self.conv_flat = ConvFlat()
        # downsample
        self.downconv_1 = ConvDown(in_channels=64, out_channels=128)
        self.downconv_2 = ConvDown(in_channels=128, out_channels=256)
        # 8 residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock() for i in range(8)])
        # upsample
        self.upconv_1 = ConvUp(in_channels=256, out_channels=128)
        self.upconv_2 = ConvUp(in_channels=128, out_channels=64)
        # back to regular images
        self.conv_to_regular_image = nn.Conv2d(in_channels=64,
                                               out_channels=3,
                                               kernel_size=7,
                                               stride=1,
                                               padding=3)

    def forward(self, x):
        return nn.Sequential(*[
            self.conv_flat,
            self.downconv_1,
            self.downconv_2,
            self.res_blocks,
            self.upconv_1,
            self.upconv_2,
            self.conv_to_regular_image
        ])(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3,
                                out_channels=32,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv_2 = nn.Conv2d(in_channels=32,
                                out_channels=64,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        self.conv_3 = nn.Conv2d(in_channels=64,
                                out_channels=128,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv_4 = nn.Conv2d(in_channels=128,
                                out_channels=128,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        self.conv_5 = nn.Conv2d(in_channels=128,
                                out_channels=256,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv_6 = nn.Conv2d(in_channels=256,
                                out_channels=256,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv_7 = nn.Conv2d(in_channels=256,
                                out_channels=1,
                                kernel_size=3,
                                stride=1,
                                padding=1)

    def forward(self, x):
        lrelu = TNF.leaky_relu
        return nn.Sequential(*[
            self.conv_1, lrelu,
            self.conv_2, lrelu,
            self.conv_3, nn.BatchNorm2d(128), lrelu,
            self.conv_4, lrelu,
            self.conv_5, nn.BatchNorm2d(256), lrelu,
            self.conv_6, nn.BatchNorm2d(256), lrelu,
            self.conv_7])(x)

if __name__ == '__main__':
    print("import this instead")
