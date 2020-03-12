import torch
from PIL import Image
import numpy as np
import os

from typing import Callable, List

from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor


def im_read(im_name):
    return np.array(Image.open(im_name))


def seg_read(seg_name):
    seg = np.array(Image.open(seg_name))
    return seg


def get_transforms(input_size: int):
    pre_tr = A.Compose([
        A.LongestMaxSize(input_size + 200),
        A.RandomCrop(input_size, input_size, p=0.5),
        A.Resize(input_size, input_size)
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

    return transforms


def get_carvana_names(root: str, mode: str) -> List[str]:
    names = list(map(lambda x: x.rsplit('.', 1)[0], os.listdir(root)))
    if mode == 'train':
        names, _ = train_test_split(names, test_size=0.2, random_state=42)
    else:
        _, names = train_test_split(names, test_size=0.2, random_state=42)

    return names


class CarvanaDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            names: List[str] = None,
            p_to_img: str = 'images',
            p_to_seg: str = 'masks',
            transforms: Callable = 'default',
            mode: str = 'train',
            input_size: int = 224
    ):
        if names is None:
            self.names = get_carvana_names(p_to_img, mode)
        self.names = names
        self.p_to_img = p_to_img
        self.p_to_seg = p_to_seg
        if transforms == 'default':
            self.transforms = get_transforms(input_size)
        else:
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
