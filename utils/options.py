import argparse

__all__ = ['get_args']


def boolstr(s):
    """ Defines a boolean string, which can be used for argparse.
    """
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Code for our paper titled "Progress on Climate Action: a Multilingual Machine Learning Analysis of the GlobalStocktake"')
    parser.add_argument('--num_topics', type=int, default=72, help='How many topics to extract from the dataset.')
    parser.add_argument('--load', type=boolstr, help='Use an existing and trained topic model.', default=False)
    return parser.parse_args()
