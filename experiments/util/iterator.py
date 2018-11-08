import numpy as np
from chainer.iterators import MultithreadIterator
from chainer.dataset.iterator import Iterator


class MultithreadSampleIterator(MultithreadIterator):
    def reset(self):
        super().reset()
        if self._shuffle:
            n = len(self.dataset)
            self._order = np.random.choice(n, n, True)

    def _get(self):
        batch = super()._get()
        if self.is_new_epoch:
            n = len(self.dataset)
            self._order = np.random.choice(n, n, True)
        return batch


class FastSliceIterator(Iterator):
    def __init__(self, dataset, batch_size):
        """
        Iterates over a dataset using slices, which can be efficiently
        implemented. Does not support shuffling or repeating.

        :param dataset: The dataset
        :param batch_size: The batch size
        """
        self._d = dataset
        self._n = len(dataset)
        self._b = batch_size
        self._i = 0

    def __next__(self):
        if self._i == self._n:
            raise StopIteration
        u = max(self._n, self._i + self._b)
        batch = self._d[self._i:u]
        self._i = u
        return batch
