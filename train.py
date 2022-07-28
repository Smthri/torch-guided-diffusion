import torch
from models import create_unet
from args import get_args
import yaml
from torchinfo import summary


if __name__ == '__main__':
    args = get_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    unet = create_unet(**config['MODEL'])
    with open('summary.txt', 'w') as f:
        f.write(str(summary(unet, input_data=[torch.randn(1, 3, 128, 128), torch.randint(0, 1, (1, ))])))

