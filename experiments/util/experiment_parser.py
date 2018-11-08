from argparse import ArgumentParser
from json import dump
from os import path, mkdir
from tempfile import mkdtemp


def _set_rng_seed(seed, device):
    """
    Sets the RNG seed

    :param seed: The seed
    :type seed: int

    :param device: Additional GPU device to seed (if not None)
    :type device: int|None
    """
    if seed is not None:
        import numpy
        numpy.random.seed(seed)
        if device is not None:
            import cupy
            cupy.random.seed(seed)


def _record_arguments(args):
    """
    Records the namespace from argparse as a json file

    :param args: The namespace object from the parsed arguments, which should
                 contain an 'output' variable to a folder to store information
                 about the experiment (e.g. the namespace returned from parsing
                 experiments.util.ExperimentParser)
    :type args: argparse.Namespace
    """
    dictionary = vars(args)
    if not path.exists(dictionary['output']):
        mkdir(dictionary['output'])
    with open(path.join(dictionary['output'], 'args'), 'w') as f:
        dump(dictionary, f)


class ExperimentParser(ArgumentParser):
    """
    Argument parser with arguments shared across all experiments
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--iterations", type=int, required=False,
                          default=10000)
        self.add_argument("--train", type=str, required=True)
        self.add_argument("--train_sample", type=int, required=False,
                          default=None)

        self.add_argument("--test", type=str, required=False, default=None)
        self.add_argument("--test_sample", type=int, required=False,
                          default=None)

        self.add_argument("--vali", type=str, required=False, default=None)
        self.add_argument("--vali_sample", type=int, required=False,
                          default=None)

        self.add_argument("--batch_size", type=int, required=False, default=1)
        self.add_argument("--eval_batch_size", type=int, required=False,
                          default=500)

        self.add_argument("--cache", type=str, required=False, default=None)
        self.add_argument("--lr", type=float, required=False, default=0.01)
        self.add_argument("--lr_decay", type=float, required=False,
                          default=None)
        self.add_argument("--l2", type=float, required=False, default=None)
        self.add_argument("--seed", type=int, required=False, default=None)
        self.add_argument("--device", type=int, required=False, default=None)
        self.add_argument("--output", type=str, required=False,
                          default=mkdtemp())
        self.add_argument("--only_eval_end", action='store_true')
        self.add_argument("--print", type=str, required=False, default=None)
        self.add_argument("--progress", action='store_true')
        self.add_argument("--save", action='store_true')

    def parse_args(self, args=None, namespace=None):
        result = super().parse_args(args, namespace)
        _record_arguments(result)
        _set_rng_seed(result.seed, result.device)
        return result
