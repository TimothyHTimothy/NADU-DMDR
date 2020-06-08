import functools
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil


class UpsampleNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=24, nu=3, upscale=4):
        super(UpsampleNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(mutil.ResidualCABlock_noBN, nf=nf)
        self.first_trunk = mutil.make_layer(basic_block, nb // 2)
        self.second_trunk = mutil.make_layer(basic_block, nb // 2)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upcon_trunk = mutil.make_layer(basic_block, nu)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.conv_medium = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last, self.conv_medium], 0.1)
        if self.upscale == 4:
            mutil.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):

        out_list = []

        # short skip connection (SSC)
        fea = self.lrelu(self.conv_first(x))
        first_fea = self.first_trunk(fea)
        # first_fea += fea

        out = self.second_trunk(first_fea)
        # out += first_fea

        # long skip connection (LSC)
        out += fea

        if self.upscale == 4:
            ## Modified ##
            ## Multi-scale upsampler ##
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            medium = self.conv_medium(self.lrelu(self.HRconv(out)))
            medium += F.interpolate(x, scale_factor=2, mode='bilinear')
            out_list.append(medium)

            identity = out
            out = self.upcon_trunk(out)
            out += identity
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        out += F.interpolate(x, scale_factor=self.upscale, mode='bilinear')
        out_list.append(out)
        return out_list


# for Dual Learning #
    
class DownsampleNet(nn.Module):
    def __init__(self, in_nc = 3, out_nc = 3, nf = 64, nb=4, downscale = 4):
        super(DownsampleNet, self).__init__()
        self.downscale = downscale
        if self.downscale == 4:
            self.conv_first = nn.Conv2d(in_nc, nf, 3, 2, 1, bias=True) 
        else:
            self.conv_first = nn.Conv2d(in_nc, nf, 3, downscale, 1, bias=True)
        basic_block = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
        self.trunk = mutil.make_layer(basic_block, nb)
        self.LR_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True) 
        if self.downscale == 4:
            self.conv_last = nn.Conv2d(nf, out_nc, 3, 2, 1, bias=True) 
        else:
            self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        mutil.initialize_weights([self.conv_first, self.conv_last], 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))

        out = self.trunk(fea)

        out += fea

        out = self.conv_last(self.lrelu(self.LR_conv(out)))
        out += F.interpolate(x, scale_factor = 1 / self.downscale, mode = 'bicubic')
        return out



