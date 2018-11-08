from __future__ import division

from math import sqrt

from chainer.training import extension


class SqrtShift(extension.Extension):

    """Trainer extension to perform a sqrt shift on an optimizer attribute.
    This extension performs f(t) = init / sqrt(t) which decreases the specified
    attribute of the optimizer. The typical use case is a decay of the learning
    rate.
    This extension is also called before the training loop starts by default.
    Args:
        attr (str): Name of the attribute to shift.
        init (float): Initial value of the attribute. If it is ``None``, the
            extension extracts the attribute at the first call and uses it as
            the initial value.
        optimizer (~chainer.Optimizer): Target optimizer to adjust the
            attribute. If it is ``None``, the main optimizer of the updater is
            used.
    """

    def __init__(self, attr, init=None, optimizer=None):
        self._attr = attr
        self._init = init
        self._optimizer = optimizer

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        if self._init is None:
            self._init = getattr(optimizer, self._attr)

    def __call__(self, trainer):
        t = float(trainer.updater.iteration)
        optimizer = self._get_optimizer(trainer)
        value = self._init / sqrt(t)
        self._update_value(optimizer, value)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
