from chainer.dataset import convert
from chainerltr.dataset import zeropad_concat


class ZeropadAsync(convert.ConcatWithAsyncTransfer):
    def __call__(self, batch, device=None, padding=None):
        if device is not None:
            return super().__call__(batch, device, 0.0)
        else:
            return zeropad_concat(batch, device)
