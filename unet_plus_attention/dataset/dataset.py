import torch
from PIL import Image
import numpy as np
import os

import albumentations as A
from albumentations.pytorch import ToTensor


pre_tr = A.Compose([
    A.LongestMaxSize(800),
    A.RandomCrop(512, 512, p=0.5),
    A.Resize(512, 512)
])

post_tr = A.Compose([
    A.Normalize(),
    ToTensor()
])


def transforms(image, mask):
    pre = pre_tr(image=image, mask=mask)
    image = post_tr(image=pre['image'])['image']
    mask = torch.from_numpy(pre['mask'])
    return image, mask.type(torch.long)


def im_read(im_name):
    return np.array(Image.open(im_name))


def seg_read(seg_name):
    seg = np.array(Image.open(seg_name))
    return seg


class CarvanaDataset(torch.utils.data.Dataset):
    def __init__(self, names, p_to_img, p_to_seg, transforms=transforms):
        self.names = names
        self.p_to_img = p_to_img
        self.p_to_seg = p_to_seg
        self.transforms = transforms

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        image_path = os.path.join(self.p_to_img, f'{name}.jpg')
        seg_path = os.path.join(self.p_to_seg, f'{name}_mask.gif')

        image = im_read(image_path)
        mask = seg_read(seg_path)
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask
