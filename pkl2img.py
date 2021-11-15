import torch
from torchvision.utils import save_image


a = torch.load('fixed_set/t_fixed_target.pkl')
save_image((a+1)/2*255, 'fixed_set/t_fixed_target.png', pad_value=255)