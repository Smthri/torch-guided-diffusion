import pytorch_lightning as pl
import torch
import wandb
import numpy as np
from skimage.io import imsave
from torchvision.utils import make_grid
from pathlib import Path
import torch.nn.functional as F
from models import ResNet50
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import yaml


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, model, diffusion, image_size, config, ckpt_folder='checkpoints', log_folder='tmp'):
        super().__init__()
        self.image_size = image_size
        self.model = model
        self.diffusion = diffusion
        self.config = config
        self.epoch = 0
        self.min_loss = np.inf
        self.ckpt_folder = Path(ckpt_folder)
        self.experiment_folder = Path(datetime.now().strftime('%d_%m_%Y__%H_%M_%S'))
        (self.ckpt_folder / self.experiment_folder).mkdir(parents=True, exist_ok=True)
        with open(str(self.ckpt_folder / self.experiment_folder / 'config.yml'), 'w') as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)
        self.log_folder = Path(log_folder)
        wandb.init(project='PyTorch-Diffusion', config=config)

    def forward(self, x):
        sampled = self.diffusion.p_sample_loop(
            self.model,
            (32, 3, self.image_size, self.image_sizeim),
            model_kwargs={'y': torch.randint(0, 9, (32,), device=self.device)},
            progress=True
        )
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

        return loss

    def training_step_end(self, loss):
        for k in loss.keys():
            loss[k] = loss[k].mean()
        return loss

    def training_epoch_end(self, outputs):
        losses = {}
        for k in outputs[0].keys():
            losses[k] = np.mean([l[k].cpu().item() for l in outputs])
        self.log('loss', losses['loss'])
        self.log('mse', losses['mse'])
        loss = losses['loss']

        AdvDis = ResNet50()
        AdvDis.to(self.device)
        path = '/srv/fast1/n.lokshin/checkpoints/100k_clean_90_64x64.pth'
        AdvDis.load_state_dict(torch.load(path, map_location=self.device))
        AdvDis.eval()

        def cond_fn(x, t, y):
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                AdvDis.zero_grad()
                logits = AdvDis(x_in)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return torch.autograd.grad(selected.sum(), x_in)[0]

        sampled = self.diffusion.p_sample_loop(
            self.model,
            (32, 3, self.image_size, self.image_size),
            model_kwargs={'y': torch.randint(0, 9, (32,), device=self.device)},
            cond_fn=None,
            progress=True
        )
        sampled = (sampled + 1) * 127.5
        sampled = torch.clamp(sampled, 0, 255)
        grid = (np.transpose(make_grid(sampled).cpu().numpy(), (1, 2, 0))).astype(np.uint8)
        imsave(str(self.log_folder / f'test-{self.epoch}_regular.png'), grid)

        sampled = self.diffusion.p_sample_loop(
            self.model,
            (32, 3, self.image_size, self.image_size),
            model_kwargs={'y': torch.randint(0, 9, (32,), device=self.device)},
            cond_fn=cond_fn,
            progress=True
        )
        sampled = (sampled + 1) * 127.5
        sampled = torch.clamp(sampled, 0, 255)
        grid = (np.transpose(make_grid(sampled).cpu().numpy(), (1, 2, 0))).astype(np.uint8)
        imsave(str(self.log_folder / f'test-{self.epoch}_guided.png'), grid)

        wandb.log(losses)
        wandb.log({'generated_images': wandb.Image(grid, caption='guided')}, step=self.epoch)

        to_save = {
            'state_dict': self.model.state_dict()
        }
        torch.save(to_save, str(self.ckpt_folder / self.experiment_folder / f'best_epoch{self.epoch}.pth'))
        if self.min_loss > loss:
            print(f'Loss decreased from {self.min_loss:.3f} to {loss:.3f}')
            to_save = {
                'state_dict': self.model.state_dict()
            }
            torch.save(to_save, str(self.ckpt_folder / f'best_epoch{self.epoch}.pth'))
            self.min_loss = loss
        self.epoch += 1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config['RUN']['lr']))
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.5
        )
        return [optimizer], {
            "scheduler": scheduler,
            "monitor": "loss"
        }
