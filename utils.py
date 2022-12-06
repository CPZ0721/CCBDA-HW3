from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

    
class TransDataset(Dataset):
    def __init__(self, image):
        self.ids = image

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = self.ids[idx]

        return img


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t.cpu())
    return c.reshape(-1, 1, 1, 1).cuda()
    
def generate_noise(x0, t, alpha_bar):

    mean = gather(alpha_bar, t) ** 0.5 * x0
    var = 1-gather(alpha_bar, t)
    eps = torch.randn_like(x0).to(x0.device)
    return mean + (var ** 0.5) * eps, eps
  
def reverse_noise(xt, noise, t, beta, alpha, alpha_bar):
    alpha_t = gather(alpha, t)
    alpha_bar_t = gather(alpha_bar, t)
    eps_coef = (1 - alpha_t) / (1 - alpha_bar_t) ** .5
    mean = 1 / (alpha_t ** 0.5) * (xt - eps_coef * noise) # Note minus sign
    var = gather(beta, t)
    eps = torch.randn(xt.shape, device=xt.device)

    return mean + (var ** 0.5) * eps