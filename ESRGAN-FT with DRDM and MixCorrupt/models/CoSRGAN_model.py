import logging
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import *
import filters

logger = logging.getLogger('base')


class SRGANModel(BaseModel):
    def __init__(self, opt):
        super(SRGANModel, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        
        # define networks and load pretrained models
        self.netDS = networks.define_DS(opt).to(self.device)
        if opt['dist']:
            self.netDS = DistributedDataParallel(self.netDS, device_ids=[torch.cuda.current_device()])
        else:
            self.netDS = DataParallel(self.netDS)

        #### added ####
        #### CO-TRAINING ####
        self.netNG = networks.define_Gene(opt).to(self.device)
        if opt['dist']:
            self.netNG = DistributedDataParallel(self.netNG, device_ids=[torch.cuda.current_device()])
        else:
            self.netNG = DataParallel(self.netNG) 

        if self.is_train:
            self.netND_LR = networks.define_Disc(opt).to(self.device)
            self.netND_HR = networks.define_D(opt).to(self.device) 
            if opt['dist']:
                self.netND_LR = DistributedDataParallel(self.netND_LR, device_ids=[torch.cuda.current_device()])
                self.netND_HR = DistributedDataParallel(self.netND_HR, device_ids=[torch.cuda.current_device()])
            else:
                self.netND_LR = DataParallel(self.netND_LR)
                self.netND_HR = DataParallel(self.netND_HR)
        #### end ####

        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)
            if opt['dist']:
                self.netD = DistributedDataParallel(self.netD,
                                                    device_ids=[torch.cuda.current_device()])
            else:
                self.netD = DataParallel(self.netD)

            self.netNG.train()
            self.netND_LR.train()
            self.netND_HR.train()
            self.netG.train()
            self.netDS.train()
            self.netD.train()

        # define losses, optimizer and scheduler
        if self.is_train:
            # ---------------------------------------- ADDED ------------------------------------------
            self.filter_low = filters.FilterLow().to(self.device)
            self.filter_high = filters.FilterHigh().to(self.device)
            self.use_filters = train_opt['use_filters']
            # -----------------------------------------------------------------------------------------

            # Noising Overall Loss
            self.cri_ng = GeneratorLoss(**vars(opt)).to(self.device)
            self.perceptual_loss = self.cri_ng.perceptual_loss
            self.l_rev_w = train_opt['reverse_weight']

            # Downsampler Dual Loss
            if train_opt['dual_weight'] > 0:
                l_pix_type = train_opt['dual_criterion']
                if l_pix_type == 'l1':
                    self.cri_dual = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_dual = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_dual_w = train_opt['dual_weight']
            else:
                logger.info('Remove dual loss.')
                self.cri_dual = None 
            
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
                self.l_lpips_w = train_opt['lpips_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                if opt['dist']:
                    self.netF = DistributedDataParallel(self.netF,
                                                        device_ids=[torch.cuda.current_device()])
                else:
                    self.netF = DataParallel(self.netF)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            self.l_gan_ref_w = train_opt['ref_weight']
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # optimizers

            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)
            # Downsample G
            wd_DS = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netDS.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_DS = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_DS,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_DS)

            # NG
            for k, v in self.netNG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            
            self.optimizer_NG = torch.optim.Adam(optim_params, lr=train_opt['lr_NG'] * self.l_rev_w,
                                                weight_decay=wd_G,
                                                betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_NG)

            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)

            # ND_HR
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_ND_HR = torch.optim.Adam(self.netND_HR.parameters(), lr=train_opt['lr_ND'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_ND_HR)

            # ND_LR
            self.optimizer_ND = torch.optim.Adam(self.netND_LR.parameters(), lr=train_opt['lr_ND'],
                                                weight_decay=wd_D,
                                                betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_ND)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        self.print_network()  # print network
        self.load()  # load networks if needed

    def feed_data(self, data, need_GT=True, is_train=True):
        if is_train is False:
            self.var_L = data['LQ'].to(self.device)
            self.noisy_F = data['NF'].to(self.device)
            if need_GT:
                self.var_H = data['GT'].to(self.device)  # GT
        else:
            self.var_L = data['LQ'].to(self.device)  # LQ
            self.noisy_F = data['NF'].to(self.device) # NF
            if need_GT:
                self.var_H = data['GT'].to(self.device)  # GT
                if self.opt['scale'] == 4:
                    self.medium_GT = F.interpolate(self.var_H, scale_factor=0.5, mode='bilinear')
                input_ref = data['ref'] if 'ref' in data else data['GT']
                self.var_ref = input_ref.to(self.device)

    def clear_data(self):
        self.var_L = None
        self.noisy_F = None
        self.var_H = None
        self.var_ref = None
        torch.cuda.empty_cache()

    
    def optimize_parameters(self, step):
        baseline=self.opt['baseline']
        if baseline:
            # G and Downsample G
            for p in self.netD.parameters():
                p.requires_grad = False

            self.optimizer_G.zero_grad()
            self.fake_H = self.netG(self.var_L)
            if self.opt['scale'] == 4:
                self.medium_H = self.fake_H[0]
            self.fake_H = self.fake_H[-1]

            self.reverse_L = self.netDS(self.fake_H)

            l_g_total = 0
            if step % self.D_update_ratio == 0 and step > self.D_init_iters:
                if self.cri_pix:  # pixel loss
                    #  ------------------------------------- ADDED ------------------------------------------
                    l_g_pix_f = self.l_pix_w * self.cri_pix(self.filter_low(self.fake_H), self.filter_low(self.var_H))
                    l_g_pix_nf = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                    if self.opt['scale'] == 4:
                        l_m_pix_f = self.l_pix_w * self.cri_pix(self.filter_low(self.medium_H), self.filter_low(self.medium_GT))
                        l_m_pix_nf = self.l_pix_w * self.cri_pix(self.medium_H, self.medium_GT)
                    # l_g_mean_color = nn.functional.mse_loss(self.fake_H.mean(3).mean(2), self.var_H.mean(3).mean(2))
                    if self.use_filters:
                        l_g_pix = l_g_pix_f
                        if self.opt['scale'] == 4 and self.opt['medium']:
                           l_g_pix += l_m_pix_f 
                    else:
                        l_g_pix = l_g_pix_nf                        
                        if self.opt['scale'] == 4 and self.opt['medium']:
                           l_g_pix += l_m_pix_f 
                        # -----------------------------------------------------------------------------------
                    # l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                    l_g_total += l_g_pix
                if self.cri_dual:
                    l_g_dual = self.l_dual_w * self.cri_dual(self.reverse_L, self.var_L)
                    l_g_total += l_g_dual
                if self.cri_fea:  # feature loss
                    real_fea = self.netF(self.var_H).detach()
                    fake_fea = self.netF(self.fake_H)
                    l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                    l_g_lpips_fea = self.l_lpips_w * self.perceptual_loss(self.fake_H, self.var_H)
                    l_g_total += l_g_fea + l_g_lpips_fea

                # ------------------------------------------ ADDED ------------------------------------------
                if self.use_filters:
                    pred_g_fake = self.netD(self.filter_high(self.fake_H))
                else:
                    # ---------------------------------------------------------------------------------------
                    pred_g_fake = self.netD(self.fake_H)
                if self.opt['train']['gan_type'] == 'gan':
                    l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
                elif self.opt['train']['gan_type'] == 'ragan':
                    # --------------------------------------- ADDED -----------------------------------------
                    if self.use_filters:
                        pred_d_real = self.netD(self.filter_high(self.var_ref)).detach()
                    else:
                        # -----------------------------------------------------------------------------------
                        pred_d_real = self.netD(self.var_ref).detach()
                    l_g_gan = self.l_gan_w * (
                        self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                        self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
                l_g_total += l_g_gan

                l_g_total.backward()
                self.optimizer_G.step()
                self.optimizer_DS.step()

                mse = ((self.var_ref - self.fake_H) ** 2).mean().data
                psnr = -10 * torch.log10(mse)
                self.log_dict['psnr'] = psnr.item()

            # D
            for p in self.netD.parameters():
                p.requires_grad = True

            self.optimizer_D.zero_grad()
            l_d_total = 0
            # ------------------------------------------ ADDED ------------------------------------------
            if self.use_filters:
                pred_d_real = self.netD(self.filter_high(self.var_ref))
                pred_d_fake = self.netD(self.filter_high(self.fake_H.detach()))  # detach to avoid BP to G
            else:
                # ---------------------------------------------------------------------------------------
                pred_d_real = self.netD(self.var_ref)
                pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
            if self.opt['train']['gan_type'] == 'gan':
                l_d_real = self.cri_gan(pred_d_real, True)
                l_d_fake = self.cri_gan(pred_d_fake, False)
                l_d_total = l_d_real + l_d_fake
            elif self.opt['train']['gan_type'] == 'ragan':
                l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
                l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
                l_d_total = l_d_real + l_d_fake

            l_d_total.backward()
            self.optimizer_D.step()

            # set log
            if step % self.D_update_ratio == 0 and step > self.D_init_iters:
                if self.cri_pix:
                    self.log_dict['l_g_pix_f'] = l_g_pix_f.item()
                    self.log_dict['l_g_pix_nf'] = l_g_pix_nf.item()
                    # self.log_dict['l_g_mean_color'] = l_g_mean_color.item()
                if self.cri_fea:
                    self.log_dict['l_g_fea'] = l_g_fea.item()
                self.log_dict['l_g_gan'] = l_g_gan.item()

            self.log_dict['l_d_real'] = l_d_real.item()
            self.log_dict['l_d_fake'] = l_d_fake.item()
            self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
            self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())    





        ## Implementations on DRDM algorithms
        else:
            for p in self.netND_LR.parameters():
                p.requires_grad = False

            self.noisy_L = self.netNG(self.var_L)

            for p in self.netD.parameters():
                p.requires_grad = False
            
            for p in self.netND_HR.parameters():
                p.requires_grad = False
            
            # G
            # -----------------------------------------------------------------------------------------------
            self.optimizer_G.zero_grad()
            self.fake_H = self.netG(self.noisy_L)
            self.noisy_H = self.netG(self.noisy_F)

            l_g_total = 0
            if step % self.D_update_ratio == 0 and step > self.D_init_iters:
                if self.cri_pix:  # pixel loss
                    #  ------------------------------------- ADDED ------------------------------------------
                    l_g_pix_f = self.l_pix_w * self.cri_pix(self.filter_low(self.fake_H), self.filter_low(self.var_H))
                    l_g_pix_nf = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                    # l_g_mean_color = nn.functional.mse_loss(self.fake_H.mean(3).mean(2), self.var_H.mean(3).mean(2))
                    if self.use_filters:
                        l_g_pix = l_g_pix_f
                    else:
                        l_g_pix = l_g_pix_nf
                        # -----------------------------------------------------------------------------------
                    # l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                    l_g_total += l_g_pix
                if self.cri_fea:  # feature loss
                    real_fea = self.netF(self.var_H).detach()
                    fake_fea = self.netF(self.fake_H)
                    l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                    if self.use_filters:
                        l_g_lpips_fea = self.l_fea_w * self.perceptual_loss(self.filter_high(self.fake_H), self.filter_high(self.var_H))
                    else:
                        l_g_lpips_fea = self.l_fea_w * self.perceptual_loss(self.fake_H, self.var_H)
                    l_g_total += l_g_fea + l_g_lpips_fea

                # ------------------------------------------ ADDED ------------------------------------------
                if self.use_filters:
                    pred_g_fake = self.netD(self.filter_high(self.fake_H))
                    pred_g_rref = self.netND_HR(self.filter_high(self.fake_H))
                    pred_g_fref = self.netND_HR(self.filter_high(self.noisy_H))
                else:
                    # ---------------------------------------------------------------------------------------
                    pred_g_fake = self.netD(self.fake_H)
                    pred_g_rref = self.netND_HR(self.fake_H)
                    pred_g_fref = self.netND_HR(self.noisy_H)
                if self.opt['train']['gan_type'] == 'gan':
                    l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
                    l_r_gan = self.l_gan_ref_w * self.cri_gan(pred_g_fref, True)
                elif self.opt['train']['gan_type'] == 'ragan':
                    # --------------------------------------- ADDED -----------------------------------------
                    if self.use_filters:
                        pred_d_real = self.netD(self.filter_high(self.var_ref)).detach()
                    else:
                        # -----------------------------------------------------------------------------------
                        pred_d_real = self.netD(self.var_ref).detach()
                    l_g_gan = self.l_gan_w * (
                        self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                        self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
                    l_r_gan = self.l_gan_ref_w * (
                        self.cri_gan(pred_g_rref - torch.mean(pred_g_fref), False) +
                        self.cri_gan(pred_g_fref - torch.mean(pred_g_rref), True)) / 2
                l_g_total += l_g_gan
                l_g_total.backward(retain_graph=True)
                l_r_gan.backward(retain_graph=True)
                self.optimizer_G.step()
                
                mse = ((self.var_ref - self.fake_H) ** 2).mean().data
                psnr = -10 * torch.log10(mse)
                mse_lq = ((self.noisy_L - self.var_L) ** 2).mean().data
                psnr_lq = -10 * torch.log10(mse_lq)
                self.log_dict['psnr'] = psnr.item()
                self.log_dict['psnr_lq'] = psnr_lq.item()

            # D & ND_HR
            # -------------------------------------------------------------------------------------------

            for p in self.netD.parameters():
                p.requires_grad = True
            if self.opt['train']['DRDM'] == True:
                for p in self.netND_HR.parameters():
                    p.requires_grad = True
            self.optimizer_D.zero_grad()
            if self.opt['train']['DRDM'] == True:
                self.optimizer_ND_HR.zero_grad()
            l_d_total = 0

            # ------------------------------------------ ADDED ------------------------------------------
            if self.use_filters:
                pred_d_real = self.netD(self.filter_high(self.var_ref))
                pred_d_fake = self.netD(self.filter_high(self.fake_H.detach()))  # detach to avoid BP to G
                if self.opt['train']['DRDM'] == True: 
                    pred_d_rref = self.netND_HR(self.filter_high(self.fake_H.detach()))
                    pred_d_fref = self.netND_HR(self.filter_high(self.noisy_H.detach()))   # detach to avoid BP to G
            else:
                # ---------------------------------------------------------------------------------------
                pred_d_real = self.netD(self.var_ref)
                pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
                if self.opt['train']['DRDM'] == True:
                    pred_d_rref = self.netND_HR(self.fake_H.detach())
                    pred_d_fref = self.netND_HR(self.noisy_H.detach())   # detach to avoid BP to G
            if self.opt['train']['gan_type'] == 'gan':
                l_d_real = self.cri_gan(pred_d_real, True)
                l_d_fake = self.cri_gan(pred_d_fake, False)
                if self.opt['train']['DRDM'] == True:
                    l_d_fref = self.cri_gan(pred_d_fref, False)
                l_d_total = l_d_real + l_d_fake
            elif self.opt['train']['gan_type'] == 'ragan':
                l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
                l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
                l_d_total = l_d_real + l_d_fake
                if self.opt['train']['DRDM'] == True:
                    l_d_fref = self.cri_gan(pred_d_fref - torch.mean(pred_d_rref), False)


            l_d_total.backward()
            self.optimizer_D.step()
            if self.opt['train']['DRDM'] == True:
                l_d_fref.backward()
                self.optimizer_ND_HR.step()

            torch.cuda.empty_cache()
            
            if self.opt['need_reverse']:
                for p in self.netND_LR.parameters():
                    p.requires_grad = True
                self.optimizer_ND.zero_grad()
                #print(self.noisy_F.shape, self.noisy_L.shape)
                if self.opt['train']['gan_type'] == 'ragan':
                    self.real_tex = self.netND_LR(self.noisy_F, self.noisy_L.detach())
                    self.fake_tex = self.netND_LR(self.noisy_L.detach(), self.noisy_F)
                else:
                    self.real_tex = self.netND_LR(self.noisy_F)
                    self.fake_tex = self.netND_LR(self.noisy_L.detach())
                
                #torch.empty_cache()
                ## ND
                l_lr_gan = discriminator_loss(self.real_tex, self.fake_tex)
                l_lr_gan.backward(retain_graph=True)
                self.optimizer_ND.step()

                ## NG
                #### Co-Training #####
                if step % self.D_update_ratio == 0 and step > self.D_init_iters:
                    #self.noisy_tex = self.netND(self.noisy_L, self.noisy_F)
                    l_ng = self.cri_ng(self.real_tex, self.noisy_L, self.var_L)
                    #self.optimizer_NG.zero_grad()
                    l_ng.backward()
                    self.optimizer_NG.step()
                    self.optimizer_NG.zero_grad()
                    
            
            # set log
            if step % self.D_update_ratio == 0 and step > self.D_init_iters:
                if self.cri_pix:
                    self.log_dict['l_g_pix_f'] = l_g_pix_f.item()
                    self.log_dict['l_g_pix_nf'] = l_g_pix_nf.item()
                    # self.log_dict['l_g_mean_color'] = l_g_mean_color.item()
                if self.cri_fea:
                    self.log_dict['l_g_fea'] = l_g_fea.item()
                self.log_dict['l_g_gan'] = l_g_gan.item()
                if self.opt['reverse']:
                    self.log_dict['real_tex'] = self.real_tex.mean().item()
                    self.log_dict['fake_tex'] = self.fake_tex.mean().item()
                    self.log_dict['l_ng'] = l_ng.item()
                    self.log_dict['l_nd'] = l_nd_gan.item()
                    #self.log_dict['psnr'] = psnr.item()
                    self.log_dict['Perceptual_Loss'] = self.cri_ng.perceptual_loss(self.var_L, self.noisy_L).item()
                    #self.log_dict['D_noisy_fake'] = self.noisy_tex.item()

            self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
            self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())
            if self.opt['train']['DRDM']:
                self.log_dict['HR_rref'] = torch.mean(pred_d_rref.detach())
                self.log_dict['HR_fref'] = torch.mean(pred_d_fref.detach()) 

            torch.cuda.empty_cache()

    def eval(self):
        self.netG.eval()
        self.netNG.eval()
    
    def test(self):
        only_SR = self.opt['baseline']
        self.eval()
        with torch.no_grad():
            if only_SR:
                self.noisy_L = self.var_L
                self.fake_H = self.netG(self.var_L)[-1]
                self.noisy_H = self.netG(self.noisy_F)[-1]
            else:
                self.noisy_L = self.netNG(self.var_L)
                self.noisy_H = self.netG(self.noisy_F)[-1]
                self.fake_H = self.netG(self.noisy_L)[-1]
        self.netG.train()
        self.netNG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True, only_SR=False):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        if only_SR is False:
            out_dict['NR'] = self.noisy_L.detach()[0].float().cpu()
            out_dict['NF'] = self.noisy_F.detach()[0].float().cpu()
            out_dict['NH'] = self.noisy_H.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()

        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel) or isinstance(
                        self.netF, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)

    def load(self):
        load_path_NG = self.opt['path']['pretrain_model_NG']
        if load_path_NG is not None:
            logger.info('Loading model for NG [{:s}] ...'.format(load_path_NG))
            self.load_network(load_path_NG, self.netNG, subpath=None)
        load_path_ND = self.opt['path']['pretrain_model_ND']
        if self.opt['is_train'] and load_path_ND is not None:
            logger.info('Loading model for ND [{:s}] ...'.format(load_path_ND))
            self.load_network(load_path_ND, self.netND_LR, subpath=None)
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])

    def save(self, iter_step):
        self.save_network(self.netNG, 'NG', iter_step)
        self.save_network(self.netND_HR, 'ND_HR', iter_step) 
        self.save_network(self.netND_LR, 'ND_LR', iter_step)
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
