import torch
from torchvision import transforms
from models import create_unet
from args import get_args
import yaml
from dataset import ConcatDatasets
import diffusion
import misc
from torch.optim import Adam
from pathlib import Path
from tqdm import tqdm


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=.5, std=.5)
    ])


if __name__ == '__main__':
    args = get_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Init the dataset and balance it
    dataset = ConcatDatasets(args.data_dir, transform=get_train_transforms())
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
    unet = create_unet(**config['MODEL'])

    timesteps = config['DIFFUSION']['diffusion_steps']
    betas = diffusion.cosine_beta_schedule(timesteps)
    gaussian_diffusion = diffusion.GaussianDiffusion(betas)

    optimizer = Adam(unet.parameters(), lr=float(config['RUN']['lr']))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    unet.to(device)

    def num_to_groups(num, divisor):
        groups = num // divisor
        remainder = num % divisor
        arr = [divisor] * groups
        if remainder > 0:
            arr.append(remainder)
        return arr

    results_folder = Path("./results")
    results_folder.mkdir(exist_ok = True)
    save_and_sample_every = 5

    # Train loop
    for epoch in range(config['RUN']['epochs']):
        losses = []

        pbar = tqdm(enumerate(train_loader), desc=f'Epoch {epoch}', total = len(dataset) // config['RUN']['batch_size'] + 1)
        for step, batch in pbar:
            optimizer.zero_grad()

            imgs, labels = batch
            batch_size = imgs.shape[0]
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = gaussian_diffusion.p_losses(
                unet, imgs, t, labels,
                loss_type="huber"
            )

            losses.append(loss.item())
            pbar.set_description(f'loss: {loss.item():.2f}')

            loss.backward()
            optimizer.step()

        print(f'Mean loss: {torch.mean(losses):.3f}')
        # save generated images
        if epoch != 0 and epoch % save_and_sample_every == 0:
            y = torch.randint(0, len(dataset.classes), (batch_size, ), device=device)
            sampled = gaussian_diffusion.sample(unet, 128, batch_size, 3, timesteps, y)
            misc.save_images(results_folder, sampled, y, dataset.classes, f'sample-{step // save_and_sample_every}.png')

