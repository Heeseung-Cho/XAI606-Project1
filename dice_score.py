from PIL import Image
import numpy as np
def compute_dice(label_img, pred_img, p_threshold=0.5):
    p = pred_img.cpu().detach().numpy()
    l = label_img.cpu().detach().numpy()
    if p.max() > 127:
        p /= 255.
    if l.max() > 127:
        l /= 255.

    p = np.clip(p, 0, 1.0)
    l = np.clip(l, 0, 1.0)
    p[p > 0.5] = 1.0
    p[p < 0.5] = 0.0
    l[l > 0.5] = 1.0
    l[l < 0.5] = 0.0
    product = np.dot(l.flatten(), p.flatten())
    dice_num = 2 * product + 1
    pred_sum = p.sum()
    label_sum = l.sum()
    dice_den = pred_sum + label_sum + 1
    dice_val = dice_num / dice_den
    return dice_val