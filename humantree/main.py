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

    args = parser.parse_args()

    # print(ht.find_poly(args.address))

    ht.get_poly_images(limit=5, purge=args.purge)

    ht.train()

    ht.predict_test()
