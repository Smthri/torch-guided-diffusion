import torch
from torchvision import transforms
from models import create_unet
from args import get_args
import yaml
from dataset import ConcatDatasets


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


if __name__ == '__main__':
    args = get_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create a backbone model
    unet = create_unet(**config['MODEL'])

    # Init the dataset and balance it
    dataset = ConcatDatasets(args.data_dir, transform=get_train_transforms())
    class_labels, class_weights = dataset.get_calibration_data()
    train_weighted_sampler = torch.utils.data.WeightedRandomSampler(
        weights=class_weights[class_labels],
        num_samples=len(dataset)
    )
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size,
                                               sampler=train_weighted_sampler,
                                               num_workers=32)

