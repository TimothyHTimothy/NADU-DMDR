from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import utils



class TrainDataset(Dataset):
    def __init__(self, noisy_dir, crop_size, target_dir=None, g_back=False, upscale_factor=4, cropped=True, flips=False, rotations=False, **kwargs):
        super(TrainDataset, self).__init__()
        # get all directories used for training
        if isinstance(noisy_dir, str):
            noisy_dir = [noisy_dir]
        self.filesX = []
        
        for n_dir in noisy_dir:
            self.filesX += [join(n_dir, x) for x in listdir(n_dir) if utils.is_image_file(x)]
        
        # moves both in target domain and source domain SR
        if isinstance(target_dir, str):
            target_dir = [target_dir]
        if target_dir is not None:
            self.need_target = True
            self.filesY = []
            for t_dir in target_dir:
                self.filesY += [join(t_dir, y) for y in listdir(t_dir) if utils.is_image_file(y)]
        else:
            self.need_target = False
            self.filesY = self.filesX
        
        print(self.filesX[0], self.filesY[0])

        # intitialize image transformations and variables
        self.input_transform = T.Compose([
            T.RandomVerticalFlip(0.25 if flips else 0.0),
            T.RandomHorizontalFlip(0.25 if flips else 0.0),
            T.RandomCrop(crop_size)
        ])
        self.crop_transform = T.RandomCrop(crop_size // upscale_factor)
        self.upscale_factor = upscale_factor
        self.cropped = cropped
        self.rotations = rotations
        self.g_back = g_back

    def __getitem__(self, index):
        # get downscaled and cropped image (if necessary), mention difference in TDSR and SDSR
        indexX = index
        gt_image = self.input_transform(Image.open(random.choice(self.filesY)))
        
        if self.rotations:
            angle = random.choice([0, 90, 180, 270])
            gt_image = TF.rotate(gt_image, angle)
        gt_image = TF.to_tensor(gt_image)
        resized_image = utils.imresize(gt_image, 1.0 / self.upscale_factor, True)
        #generate noisy image
        if self.cropped:
            noisy_image = self.input_transform(Image.open(self.filesX[indexX]))
            if self.rotations:
                angle = random.choice([0, 90, 180, 270])
                noisy_image = TF.rotate(noisy_image, angle) 
            cropped_image = self.crop_transform(noisy_image)
            if self.g_back:
                return TF.to_tensor(cropped_image), resized_image
            else:
                #print('is_here')
                return resized_image, TF.to_tensor(cropped_image)
        else:
            return resized_image

    def __len__(self):
        return len(self.filesX) 

class DiscDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor=4, flips=False, rotations=False, **kwargs):
        super(DiscDataset, self).__init__()
        self.filesX = [join(dataset_dir, x) for x in listdir(dataset_dir) if utils.is_image_file(x)]
        self.input_transform = T.Compose([
            T.RandomVerticalFlip(0.5 if flips else 0.0),
            T.RandomHorizontalFlip(0.5 if flips else 0.0),
            T.RandomCrop(crop_size // upscale_factor)
        ])
        self.rotations = rotations

    def __getitem__(self, index):
        # get real image for discriminator (same as cropped in TrainDataset)
        image = self.input_transform(Image.open(self.files[index]))
        if self.rotations:
            angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, angle)
        return TF.to_tensor(image)

    def __len__(self):
        return len(self.files)


class ValDataset(Dataset):
    def __init__(self, hr_dir, upscale_factor, lr_dir=None, crop_size_val=None, g_back=False, **kwargs):
        super(ValDataset, self).__init__()
        self.hr_files = [join(hr_dir, x) for x in listdir(hr_dir) if utils.is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.crop_size = crop_size_val
        self.g_back = g_back
        if lr_dir is None:
            self.lr_files = None
        else:
            self.lr_files = [join(lr_dir, x) for x in listdir(lr_dir) if utils.is_image_file(x)]

    def __getitem__(self, index):
        # get downscaled, cropped and gt (if available) image
        hr_image = Image.open(self.hr_files[index])
        w, h = hr_image.size
        cs = utils.calculate_valid_crop_size(min(w, h), self.upscale_factor)
        if self.crop_size is not None:
            cs = min(cs, self.crop_size)
        cropped_image = TF.to_tensor(T.CenterCrop(cs // self.upscale_factor)(hr_image))
        hr_image = T.CenterCrop(cs)(hr_image)
        hr_image = TF.to_tensor(hr_image)
        resized_image = utils.imresize(hr_image, 1.0 / self.upscale_factor, True)
        if self.lr_files is None:
            if self.g_back:
                return cropped_image, resized_image, cropped_image
            else:
                return resized_image, cropped_image, resized_image
        else:
            lr_image = Image.open(self.lr_files[index])
            lr_image = TF.to_tensor(T.CenterCrop(cs // self.upscale_factor)(lr_image))
            return resized_image, cropped_image, lr_image

    def __len__(self):
        return len(self.hr_files)
