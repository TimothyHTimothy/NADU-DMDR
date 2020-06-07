import torch
from models import ds_model
import logging
import models.modules.SRResNet_arch as SRResNet_arch
import models.modules.discriminator_vgg_arch as SRGAN_arch
import models.modules.RRDBNet_arch as RRDBNet_arch
logger = logging.getLogger('base')


####################
# define network

####################
#### DSGAN

def define_Disc(opt):
    opt_d = opt['network_DS']['disc']
    netDisc = ds_model.Discriminator(recursions=opt_d['recursions'], stride=opt_d['stride'],
                                     kernel_size=opt_d['kernel_size'])
    return netDisc

def define_Gene(opt):
    opt_net = opt['network_DS']
    opt_g = opt_net['gene']
    netGene = ds_model.Generator(n_blocks=opt_g['n_blocks'], n_groups=opt_g['n_groups'],
                                 need_ca=opt_g['need_ca'], need_dense=opt_g['need_dense'], reduction=opt_g['red'])
    return netGene



####################
#### Generator

def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'MSRResNet':
        netG = SRResNet_arch.UpsampleNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], nu=opt_net['nu'], upscale=opt_net['scale'])
    elif which_model == 'RRDBNet':
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    # elif which_model == 'sft_arch':  # SFT-GAN
    #     netG = sft_arch.SFT_Net()
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG

def define_DS(opt):
    opt_net = opt['network_G']
    netDS = SRResNet_arch.DownsampleNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'],  downscale=opt_net['scale'])
    return netDS


#### Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
