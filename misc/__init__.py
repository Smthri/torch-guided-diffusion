from skimage.io import imsave
import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm


def save_tensor_as_img(fname, t, batch_idx=0):
    if len(t.shape) == 4:
        t = t[batch_idx]
    assert len(t.shape) == 3, 'Passed tensor must be either [N x C x H x W] or [C x H x W].'

    t = (t * torch.Tensor([0.5])[:, None, None]) + torch.Tensor([0.5])[:, None, None]
    t = (np.transpose(t.numpy(), (1, 2, 0)) * 255).astype(np.uint8)
    imsave('test.png', t)


def save_images(result_dir, sampled, labels, class_list, fname):
    """
    result_dir - folder to save milestones
    sampled - list of shape (T x N x C x H x W)
    """
    sampled = torch.transpose(sampled, 0, 1)
    for n in tqdm(range(len(sampled)), desc='Saving images....'):
        filepath = result_dir / class_list[labels[n]] / fname
        filepath.parent.mkdir(parents=True, exist_ok=True)
        save_image(sampled[n], str(filepath))
