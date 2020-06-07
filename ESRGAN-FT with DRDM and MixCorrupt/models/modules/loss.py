import torch
import torch.nn as nn

import torch
import random
from torch import nn
from torchvision.models.vgg import vgg16, vgg19
from models.ds_model import FilterLow
import sys
#from models.PerceptualSimilarity import models


def generator_loss(labels, wasserstein=False, weights=None):
    if not isinstance(labels, list):
        labels = (labels,)
    if weights is None:
        weights = [1.0 / len(labels)] * len(labels)
    loss = 0.0
    for label, weight in zip(labels, weights):
        if wasserstein:
            loss += weight * torch.mean(-label)
        else:
            loss += weight * torch.mean(-torch.log(label + 1e-8))
    return loss


def discriminator_loss(reals, fakes, wasserstein=False, grad_penalties=None, weights=None):
    if not isinstance(reals, list):
        reals = (reals,)
    if not isinstance(fakes, list):
        fakes = (fakes,)
    if weights is None:
        weights = [1.0 / len(fakes)] * len(fakes)
    loss = 0.0
    if wasserstein:
        if not isinstance(grad_penalties, list):
            grad_penalties = (grad_penalties,)
        for real, fake, weight, grad_penalty in zip(reals, fakes, weights, grad_penalties):
            loss += weight * (-real.mean() + fake.mean() + grad_penalty)
    else:
        for real, fake, weight in zip(reals, fakes, weights):
            loss += weight * (-torch.log(real + 1e-8).mean() - torch.log(1 - fake + 1e-8).mean())
    return loss


class GeneratorLoss(nn.Module):
    def __init__(self, recursions=1, stride=1, kernel_size=5, use_perceptual_loss=True, wgan=False, w_col=1,
                 w_tex=0.001, w_per=0.1, gaussian=False, lpips_rot_flip=False, **kwargs):
        super(GeneratorLoss, self).__init__()
        self.pixel_loss = nn.L1Loss()
        self.color_filter = FilterLow(recursions=recursions, stride=stride, kernel_size=kernel_size, padding=False,
                                      gaussian=gaussian)
        if torch.cuda.is_available():
            self.pixel_loss = self.pixel_loss.cuda()
            self.color_filter = self.color_filter.cuda()
        self.perceptual_loss = PerceptualLoss(rotations=lpips_rot_flip, flips=lpips_rot_flip)
        self.use_perceptual_loss = use_perceptual_loss
        self.wasserstein = wgan
        self.w_col = w_col
        self.w_tex = w_tex
        self.w_per = w_per
        self.last_tex_loss = 0
        self.last_per_loss = 0
        self.last_col_loss = 0
        self.last_mean_loss = 0

    def forward(self, tex_labels, out_images, target_images):
        # Adversarial Texture Loss
        self.last_tex_loss = generator_loss(tex_labels, wasserstein=self.wasserstein)
        # Perception Loss
        self.last_per_loss = self.perceptual_loss(out_images, target_images)
        # Color Loss
        self.last_col_loss = self.color_loss(out_images, target_images)
        loss = self.w_col * self.last_col_loss + self.w_tex * self.last_tex_loss
        if self.use_perceptual_loss:
            loss += self.w_per * self.last_per_loss
        return loss

    def color_loss(self, x, y):
        return self.pixel_loss(self.color_filter(x), self.color_filter(y))

    def rgb_loss(self, x, y):
        return self.pixel_loss(x.mean(3).mean(2), y.mean(3).mean(2))

    def mean_loss(self, x, y):
        return self.pixel_loss(x.view(x.size(0), -1).mean(1), y.view(y.size(0), -1).mean(1))



class PerceptualLossVGG16(nn.Module):
    def __init__(self):
        super(PerceptualLossVGG16, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, x, y):
        return self.mse_loss(self.loss_network(x), self.loss_network(y))


class PerceptualLossVGG19(nn.Module):
    def __init__(self):
        super(PerceptualLossVGG19, self).__init__()
        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:36]).eval()
        if torch.cuda.is_available():
            loss_network = loss_network.cuda()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, x, y):
        return self.mse_loss(self.loss_network(x), self.loss_network(y))



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
        
class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss
