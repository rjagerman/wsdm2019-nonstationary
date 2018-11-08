from chainer import as_variable, cuda, Chain


class Corrupt(Chain):
    def __init__(self, indices):
        """
        Corrupts by setting features at specific indices to zero

        :param indices: The indices
        :type indices: list
        """
        super().__init__()
        self.indices = indices
        self._cache = {}

    def __call__(self, x):
        xp = cuda.get_array_module(x)
        if x.shape[1] not in self._cache:
            mask = xp.ones((1, x.shape[1]), dtype=x.dtype)
            mask[0, self.indices] = 0.0
            self._cache[x.shape[1]] = mask
        else:
            mask = self._cache[x.shape[1]]
        mask = as_variable(xp.broadcast_to(mask, x.shape))
        return x * mask
