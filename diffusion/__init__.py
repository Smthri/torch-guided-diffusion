import torch
import torch.nn.functional as F
from tqdm import tqdm
from .schedules import (
    cosine_beta_schedule,
    linear_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule
)
from .gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType
)
mean_type_dict = {
    'epsilon': ModelMeanType.EPSILON,
    'x_prev': ModelMeanType.PREVIOUS_X,
    'x_0': ModelMeanType.START_X
}
var_type_dict = {
    'learned': ModelVarType.LEARNED,
    'fixed_small': ModelVarType.FIXED_SMALL,
    'fixed_large': ModelVarType.FIXED_LARGE,
    'range': ModelVarType.LEARNED_RANGE
}
loss_type_dict = {
    'mse': LossType.MSE,
    'rescaled_mse': LossType.RESCALED_MSE,
    'kl': LossType.KL,
    'rescaled_kl': LossType.RESCALED_KL
}


def create_diffusion(
    betas, model_mean_type,
    model_var_type, loss_type, rescale_timesteps
):
    return GaussianDiffusion(
        betas,
        mean_type_dict[model_mean_type],
        var_type_dict[model_var_type],
        loss_type_dict[loss_type],
        rescale_timesteps
    )


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
