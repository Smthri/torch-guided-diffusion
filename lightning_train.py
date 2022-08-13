import torch
from torchvision import transforms
import models
from args import get_train_args
import yaml
from dataset import ConcatDatasets
import diffusion
from wrapper import DiffusionWrapper
import pytorch_lightning as pl
import sys


def get_train_transforms(imsize):
    return transforms.Compose([
        transforms.RandomCrop(200),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(imsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=.5, std=.5)
    ])


if __name__ == '__main__':
    args = get_train_args()

    if args.config is not None and args.continue_ckpt is None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Create a backbone model
        print('Creating model...')
        unet = models.create_unet(**config['MODEL'])
        print('Done.')

        # Create diffusion processor
        timesteps = config['DIFFUSION']['diffusion_steps']
        betas = diffusion.get_betas(config['DIFFUSION']['beta_schedule'], timesteps)
        gaussian_diffusion = diffusion.create_diffusion(
            betas,
            **config['DIFFUSION']
        )

        # Create timestep sampler
        sampler = diffusion.create_named_schedule_sampler(config['DIFFUSION']['schedule_sampler'], gaussian_diffusion)

        process = DiffusionWrapper(
            model=unet,
            diffusion=gaussian_diffusion,
            image_size=config['MODEL']['image_size'],
            config=config,
            sampler=sampler,
            ckpt_folder=config['RUN']['ckpt_dir'],
            log_folder=config['RUN']['log_folder']
        )
    elif args.config is None and args.continue_ckpt is not None:
        process = DiffusionWrapper.load_from_checkpoint(args.continue_ckpt)
        config = process.config
    else:
        print('Invalid combination of arguments \"continue_ckpt\" and \"config\", must pass only one.')
        sys.exit(0)

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
        persistent_workers=True,
        pin_memory=True,
    )
    config['RUN']['classes'] = dataset.classes

    trainer = pl.Trainer(
        default_root_dir=config['RUN']['ckpt_dir'],
        enable_checkpointing=True,
        max_epochs=config['RUN']['epochs'],
        gpus=-1,
        replace_sampler_ddp=False,
        accelerator='auto',
        amp_level='O3',
        amp_backend='apex',
        strategy='dp'
    )
    trainer.fit(process, train_dataloaders=train_loader)

