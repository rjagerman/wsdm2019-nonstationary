from chainer import as_variable
from chainer.backends import cuda


class PresentationBiasPropensityEstimator():
    def __init__(self, eta=1.0):
        self.eta = eta

    def __call__(self, log_p):
        """
        Simple propensity model for presentation bias

        :param log_p: The log propensities of the model
        :type log_p: chainer.Variable

        :return: The propensities according to a simple presentation bias model
        :rtype: chainer.Variable
        """
        xp = cuda.get_array_module(log_p)
        lp = xp.reshape(xp.arange(log_p.shape[1], dtype=log_p.dtype),
                        (1, log_p.shape[1]))
        lp = (1. / (1. + lp)) ** self.eta
        return as_variable(xp.log(xp.broadcast_to(lp, log_p.shape)))
