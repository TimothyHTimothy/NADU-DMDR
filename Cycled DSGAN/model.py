from torch import nn
import torch
import math


class Generator(nn.Module):
    def __init__(self, n_blocks=4, n_groups=2,scale=1, need_ca=False, need_dense=False, reduction=16):
        super(Generator, self).__init__()
        self.block_input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.n_groups = n_groups
        self.n_blocks = n_blocks
        self.scale = scale
        self.need_dense = need_dense
        if need_ca:
            self.res_blocks = nn.ModuleList([ResidualCABlock(64, reduction) for _ in range(n_blocks * n_groups)])
        else:
            self.res_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(n_blocks * n_groups)])
        self.feat_fus = nn.Conv2d((n_blocks * n_groups + 1) * 64, 64, kernel_size=1)
        if scale != 1:
            self.upsampler = Upsampler(scale, 64)
        self.block_output = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        

    def forward(self, x):
        #print("The shape of input is", x.shape)
        x = self.block_input(x)
        feat = []
        feat.append(x)
        for res_block in self.res_blocks:
            block = res_block(feat[len(feat)-1])
            # Short Skip Connection
            if len(feat) % self.n_groups == 0:
                block += feat[len(feat)-1]
            feat.append(block)
        #Feature Fusion
        if self.need_dense:
            block = torch.cat(feat, 1)
            block = self.feat_fus(block)
        # Long Skip Connection
        block = block + x
        if self.scale != 1:
            block = self.upsampler(block)
        block = self.block_output(block)
        block = torch.sigmoid(block)
        #print("The shape of the generated image is", block.shape)
        return block

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feat, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feat, 4 * n_feat, kernel_size=3, padding=1, bias=bias))
                m.append(nn.PixelShuffle(2))
                m.append(nn.Sigmoid())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)
        


class Discriminator(nn.Module):
    def __init__(self, recursions=1, stride=1, kernel_size=5, gaussian=False, wgan=False, highpass=True):
        super(Discriminator, self).__init__()
        if highpass:
            self.filter = FilterHigh(recursions=recursions, stride=stride, kernel_size=kernel_size, include_pad=False,
                                     gaussian=gaussian)
        else:
            self.filter = None
        self.net = DiscriminatorBasic(n_input_channels=3)
        self.wgan = wgan

    def forward(self, x, y=None):
        if self.filter is not None:
            x = self.filter(x)
        x = self.net(x)
        if y is not None:
            x -= self.net(self.filter(y)).mean(0, keepdim=True)
        if not self.wgan:
            x = torch.sigmoid(x)
        return x


class DiscriminatorBasic(nn.Module):
    def __init__(self, n_input_channels=3):
        super(DiscriminatorBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)

class CALayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        return x + residual

class ResidualCABlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ResidualCABlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.calayer = CALayer(channels, reduction)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.calayer(residual)
        return x + residual


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, stride=1, padding=4):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        mean = (kernel_size - 1) / 2.0
        variance = (kernel_size / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size, stride=stride, padding=padding, groups=3, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)


class FilterLow(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, padding=True, include_pad=True, gaussian=False):
        super(FilterLow, self).__init__()
        if padding:
            pad = int((kernel_size - 1) / 2)
        else:
            pad = 0
        if gaussian:
            self.filter = GaussianFilter(kernel_size=kernel_size, stride=stride, padding=pad)
        else:
            self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad, count_include_pad=include_pad)
        self.recursions = recursions

    def forward(self, img):
        for i in range(self.recursions):
            img = self.filter(img)
        return img


class FilterHigh(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, include_pad=True, normalize=True, gaussian=False):
        super(FilterHigh, self).__init__()
        self.filter_low = FilterLow(recursions=1, kernel_size=kernel_size, stride=stride, include_pad=include_pad,
                                    gaussian=gaussian)
        self.recursions = recursions
        self.normalize = normalize

    def forward(self, img):
        if self.recursions > 1:
            for i in range(self.recursions - 1):
                img = self.filter_low(img)
        img = img - self.filter_low(img)
        if self.normalize:
            return 0.5 + img * 0.5
        else:
            return img

def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        #print(requires_grad)
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad