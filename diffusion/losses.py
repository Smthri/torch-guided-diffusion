import torch.nn.functional as F


def l1(noise, predicted_noise):
    return F.l1_loss(predicted_noise, noise)


def l2(noise, predicted_noise):
    return F.mse_loss(predicted_noise, noise)


def huber(noise, predicted_noise):
    return F.smooth_l1_loss(predicted_noise, noise)
