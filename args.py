import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Utility for training diffusion models.')

    parser.add_argument(
        'data_dir',
        metavar='data_dirs',
        nargs='+',
        type=str,
        help='Path to dataset roots.'
    )
    parser.add_argument(
        '-cfg', '--config',
        type=str,
        help='Run configuration.'
    )

    return parser.parse_args()
