import torch
from torchvision import transforms
import models
from args import get_args
import yaml
from dataset import ConcatDatasets
import diffusion
import misc
from torch.optim import Adam
from pathlib import Path
from tqdm import tqdm
import wandb
import numpy as np
from torchinfo import summary
from torchvision.utils import make_grid
from skimage.io import imsave
from PIL import Image


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
    wandb.init(project='PyTorch-Diffusion', config=config)

    IMAGE_SIZE = config['MODEL']['image_size']

    # Init the dataset and balance it
    dataset = ConcatDatasets(args.data_dir, transform=get_train_transforms(IMAGE_SIZE))
    class_labels, class_weights = dataset.get_calibration_data()
    train_weighted_sampler = torch.utils.data.WeightedRandomSampler(
        weights=class_weights[class_labels],
        num_samples=len(dataset)
    )
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=config['RUN']['batch_size'],
                                               sampler=train_weighted_sampler,
                                               num_workers=32)

    # Create a backbone model
    config['MODEL']['num_classes'] = len(dataset.classes)
    print('Creating model...')
    #unet = models.Unet(
    #    dim=128,
    #    num_classes=9
    #)
    unet = models.create_unet(**config['MODEL'])
    with open('summary.txt', 'w') as f:
        f.write(str(summary(unet, input_data=[
            torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
            torch.randint(0, 1, (1,)),
            torch.randint(0, 9, (1,))
        ])))
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

    optimizer = Adam(unet.parameters(), lr=float(config['RUN']['lr']))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    unet.to(device)

    results_folder = Path("./results")
    results_folder.mkdir(exist_ok = True)
    save_and_sample_every = 1

    min_loss = np.inf

    # Train loop
    for epoch in range(config['RUN']['epochs']):
        losses = []

        print(f'Starting epoch {epoch}')
        pbar = tqdm(
            enumerate(train_loader),
            total = len(dataset) // config['RUN']['batch_size'] + 1
        )
        for step, batch in pbar:
            optimizer.zero_grad()

            imgs, labels = batch
            batch_size = imgs.shape[0]
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = gaussian_diffusion.training_losses(
                unet, imgs, t,
                model_kwargs={'y': labels}
            )

            loss_dict = {}
            for key in loss.keys():
                loss_dict[key] = loss[key].detach().mean().item()
            losses.append(loss_dict)
            pbar.set_description(str([f'{key}: {loss_dict[key]:.3f}' for key in loss_dict.keys()]))

            loss = loss['loss'].mean()
            loss.backward()
            optimizer.step()

        mean_dict = {}
        for k in losses[0].keys():
            mean_dict[k] = np.mean([l[k] for l in losses])
        print(mean_dict)
        wandb.log(mean_dict)
        if (mean_dict['loss'] < min_loss):
            print(f'Loss decreased from {min_loss:.3f} to {mean_dict["loss"]:.3f}')
            min_loss = mean_dict['loss']
            to_save = {
                'config': config,
                'state_dict': unet.state_dict()
            }
            torch.save(to_save, f'checkpoints/ckpt_epoch{epoch}.pth')
            # save generated images
        if epoch % save_and_sample_every == 0:
            sampled = gaussian_diffusion.p_sample_loop(
                unet,
                (32, 3, IMAGE_SIZE, IMAGE_SIZE),
                model_kwargs={'y': torch.randint(0, 9, (32,), device=device)},
                progress=True
            )
            sampled = sampled * 0.5 + 0.5
            grid = (np.transpose(make_grid(sampled).cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
            imsave(f'test-{epoch}.png', grid)
            wandb.log({'generated_images': wandb.Image(grid)}, step=epoch)

