from typing import List
from functools import partial

import gc
from tqdm import tqdm as tqdm

import torch


#  https://github.com/catalyst-team/catalyst/blob/master/catalyst/utils/metrics/iou.py
def iou(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    # values are discarded, only None check
    # used for compatibility with MultiMetricCallback
    classes: List[str] = None,
    eps: float = 1e-7,
    threshold: float = None,
    activation: str = "Argmax"
):
    if activation == 'Sigmoid':
        activation_fn = torch.sigmoid
    elif activation == 'Softmax2d':
        activation_fn = torch.nn.functional.softmax
    elif activation == 'Argmax':
        activation_fn = (lambda x: x)
        outputs = outputs.argmax(1)
    else:
        activation_fn = (lambda x: x)
    outputs = activation_fn(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    # ! fix backward compatibility
    if classes is not None:
        # if classes are specified we reduce across all dims except channels
        _sum = partial(torch.sum, dim=[0, 2, 3])
    else:
        _sum = torch.sum

    intersection = _sum(targets * outputs)
    union = _sum(targets) + _sum(outputs)
    # this looks a bit awkward but `eps * (union == 0)` term
    # makes sure that if I and U are both 0, than IoU == 1
    # and if U != 0 and I == 0 the eps term in numerator is zeroed out
    # i.e. (0 + eps) / (U - 0 + eps) doesn't happen
    iou_ = (intersection + eps * (union == 0)) / (union - intersection + eps)

    return iou_


def eval_iou(model, dataloader, device, verbose=True):
    gc.collect()
    torch.cuda.empty_cache()

    model.to(device)
    model.eval()
    with torch.no_grad():
        iou_score = 0
        num_objects = 0
        if verbose:
            dataloader = tqdm(dataloader)
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device).type(torch.long)
            iou_score += iou(model(images).cpu(), masks.cpu()).detach().item()
            num_objects += images.shape[0]
            del images
            del masks
            gc.collect()
            torch.cuda.empty_cache()
        score = iou_score / num_objects
    return score
