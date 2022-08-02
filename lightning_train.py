import torch
from torchvision import transforms
import models
from args import get_args
import yaml
from dataset import ConcatDatasets
import diffusion
from torchinfo import summary
from wrapper import DiffusionWrapper
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy, DDPSpawnStrategy


def get_train_transforms(imsize):
    return transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=.5, std=.5)
    ])


if __name__ == '__main__':
    args = get_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    IMAGE_SIZE = config['MODEL']['image_size']

    # Init the dataset and balance it
    dataset = ConcatDatasets(args.data_dir, transform=get_train_transforms(IMAGE_SIZE))
    class_labels, class_weights = dataset.get_calibration_data()
    train_weighted_sampler = torch.utils.data.WeightedRandomSampler(
        weights=class_weights[class_labels],
        num_samples=len(dataset)
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['RUN']['batch_size'],
        sampler=train_weighted_sampler,
        num_workers=32,
        persistent_workers=True
    )

    # Create a backbone model
    config['MODEL']['num_classes'] = len(dataset.classes)
    print('Creating model...')
    unet = models.create_unet(**config['MODEL'])
    '''with open('summary.txt', 'w') as f:
        f.write(str(summary(unet, input_data=[
            torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
            torch.randint(0, 1, (1,)),
            torch.randint(0, 9, (1,))
        ])))'''
    print('Done.')

    # Create diffusion processor
    timesteps = config['DIFFUSION']['diffusion_steps']
    betas = diffusion.cosine_beta_schedule(timesteps)
    gaussian_diffusion = diffusion.create_diffusion(
        betas,
        config['DIFFUSION']['model_mean_type'],
        config['DIFFUSION']['model_var_type'],
        config['DIFFUSION']['loss'],
        config['DIFFUSION']['rescale_timesteps']
    )

    trainer = pl.Trainer(
        max_epochs=config['RUN']['epochs'],
        gpus=-1,
        replace_sampler_ddp=False,
        accelerator='gpu',
        amp_level='O3',
        amp_backend='apex',
        strategy='dp'
    )
    process = DiffusionWrapper(
        model=unet,
        diffusion=gaussian_diffusion,
        image_size=IMAGE_SIZE,
        config=config,
        ckpt_folder=config['RUN']['ckpt_dir']
    )
    trainer.fit(process, train_dataloaders=train_loader)

