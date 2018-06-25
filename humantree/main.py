"""Command-line interface for class."""
import argparse
from operator import attrgetter

from .humantree import HumanTree


class SortingHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Sort argparse arguments by argument name."""

    def add_arguments(self, actions):
        """Add sorting action based on `option_strings`."""
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)


def main():
    """Run command-line version of HumanTree."""
    ht = HumanTree()

    parser = argparse.ArgumentParser(
        prog='humantree',
        description=(
            'Count trees on a property and make suggestions for new trees.'),
        formatter_class=SortingHelpFormatter)

    parser.add_argument(
        '--address',
        '-a',
        dest='address',
        type=str,
        default='21 Linnaean St, Cambridge, MA 02138',
        nargs='?',
        help='Address to perform operations on.')

    parser.add_argument(
        '--purge-cache',
        '-p',
        dest='purge',
        default=False,
        action='store_true',
        help='Purge cached files.')

    parser.add_argument(
        '--tasks',
        '-t',
        dest='tasks',
        type=str,
        default=['collect', 'train', 'predict'],
        nargs='*',
        help='Tasks to perform.')

    parser.add_argument(
        '--size',
        '-s',
        dest='size',
        type=int,
        default=4000,
        help='Set number of images to train/validate on.')

    args = parser.parse_args()

    # print(ht.find_poly(args.address))

    if 'collect' in args.tasks:
        ht.get_poly_images(limit=args.size, purge=args.purge)

    if 'train' in args.tasks:
        ht.train()

    if 'predict' in args.tasks:
        ht.predict(limit=args.size, kind='all')
