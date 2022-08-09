from .resample import (
    UniformSampler,
    LossSecondMomentResampler,
    LossAwareSampler
)
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
    'learned_range': ModelVarType.LEARNED_RANGE
}
loss_type_dict = {
    'mse': LossType.MSE,
    'rescaled_mse': LossType.RESCALED_MSE,
    'kl': LossType.KL,
    'rescaled_kl': LossType.RESCALED_KL
}


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


def get_betas(
    schedule_type,
    num_timesteps,
    s=0.008
):
    if schedule_type == 'cosine':
        return cosine_beta_schedule(num_timesteps, s)
    elif schedule_type == 'linear':
        return linear_beta_schedule(num_timesteps)
    elif schedule_type == 'quadratic':
        return quadratic_beta_schedule(num_timesteps)
    elif schedule_type == 'sigmoid':
        return sigmoid_beta_schedule(num_timesteps)
    else:
        raise NotImplementedError('Unknown beta schedule')


def create_diffusion(
    betas, model_mean_type,
    model_var_type, loss_type, rescale_timesteps,
    p2, p2_gamma, p2_k, **kwargs
):
    return GaussianDiffusion(
        betas,
        mean_type_dict[model_mean_type],
        var_type_dict[model_var_type],
        loss_type_dict[loss_type],
        rescale_timesteps,
        p2, p2_gamma, p2_k
    )


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
