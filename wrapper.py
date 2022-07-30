import pytorch_lightning as pl
import torch
import wandb
import numpy as np
from skimage.io import imsave
from torchvision.utils import make_grid
from pathlib import Path


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, model, diffusion, image_size, config, ckpt_folder='checkpoints'):
        super().__init__()
        self.image_size = image_size
        self.model = model
        self.diffusion = diffusion
        self.config = config
        self.epoch = 0
        self.min_loss = np.inf
        self.ckpt_folder = Path(ckpt_folder)
        if self.local_rank == 0:
            wandb.init(project='PyTorch-Diffusion', config=config, group="DDP")

    def forward(self, x):
        sampled = self.diffusion.p_sample_loop(
            self.model,
            (32, 3, self.image_size, self.image_sizeim),
            model_kwargs={'y': torch.randint(0, 9, (32,), device=self.device)},
            progress=True
        )
        sampled = sampled * 0.5 + 0.5
        return sampled

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        batch_size = imgs.shape[0]

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device).long()

        loss = self.diffusion.training_losses(
            self.model, imgs, t,
            model_kwargs={'y': labels}
        )

        loss = loss['loss'].mean()
        return loss

    def training_step_end(self, outputs):
        loss = torch.mean(outputs)
        return loss

    def training_epoch_end(self, outputs):
        loss = np.mean([l['loss'].cpu().item() for l in outputs])
        sampled = self.diffusion.p_sample_loop(
            self.model,
            (32, 3, self.image_size, self.image_size),
            model_kwargs={'y': torch.randint(0, 9, (32,), device=self.device)},
            progress=True
        )
        sampled = sampled * 0.5 + 0.5
        grid = (np.transpose(make_grid(sampled).cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
        imsave(f'test-{self.epoch}.png', grid)

        if self.local_rank == 0:
            wandb.log({'loss': loss})
            wandb.log({'generated_images': wandb.Image(grid)}, step=self.epoch)

            if self.min_loss < loss:
                print(f'Loss decreased from {self.min_loss:.3f} to {loss:.3f}')
                to_save = {
                    'config': self.config,
                    'state_dict': self.model.state_dict()
                }
                torch.save(to_save, str(self.ckpt_folder / f'ckpt_epoch{self.epoch}.pth'))
                self.min_loss = loss
            self.epoch += 1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config['RUN']['lr']))
        return optimizer
