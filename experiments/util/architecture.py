from chainer import links as L

from experiments.util.corrupt import Corrupt


class LinearArchitecture(L.Linear):
    def __init__(self, corrupt=None, output_size=1):
        super().__init__(None, output_size)
        with self.init_scope():
            self.corrupt = None if corrupt is None else Corrupt(corrupt)

    def __call__(self, x):
        out = x
        if self.corrupt is not None:
            out = self.corrupt(out)
        return super().__call__(out)
