from chainer import Chain, as_variable
from chainer.backends import cuda

from experiments.util.bandify import RankingBandify
from experiments.util.propensity_estimator import \
    PresentationBiasPropensityEstimator


class MovingAverageEvaluator(Chain):
    def __init__(self, policy_to_evaluate, logging_policy, click_model,
                 eta=1.0):
        super().__init__(policy_to_evaluate=policy_to_evaluate)
        with super().init_scope():
            self.bandify = RankingBandify(logging_policy, click_model)
            self.propensity_estimator = PresentationBiasPropensityEstimator(eta)

    def __call__(self, x, y, nr_docs):
        x, y, nr_docs = as_variable(x), as_variable(y), as_variable(nr_docs)
        x, a, log_p, r = self.bandify(x, y, nr_docs)
        #log_p = self.propensity_estimator(log_p)
        return self.evaluate(x, a, log_p, r)

    def evaluate(self, x, a, log_p, r):
        """
        Evaluates the next batch of samples

        :param x: The observations
        :type x: chainer.Variable

        :param a: The actions
        :type a: chainer.Variable

        :param log_p: The log propensity scores
        :type log_p: chainer.Variable

        :param r: The rewards
        :type r: chainer.Variable

        :return: The loss
        :rtype: chainer.Variable
        """
        pass


class ExponentialMovingStatistic:
    def __init__(self, decay):
        self.mean = 0.0
        self._biased_mean = 0.0
        self.n = 0
        self.decay = decay

    def add(self, batch):
        """
        Adds a batch of observations

        :param batch: The batch of observations
        :type batch: chainer.Variable
        """
        xp = cuda.get_array_module(batch)
        dtype = batch.dtype
        n = batch.shape[0]
        weights = self.decay ** (xp.arange(n, 0, -1.0, dtype=dtype) - 1.0)
        weighted_batch = weights * batch.data

        # Compute (bias-corrected) mean
        sum = (1 - self.decay) * xp.sum(weighted_batch, axis=0)
        self._biased_mean = (self.decay ** n) * self._biased_mean + sum
        self.n += n
        self.mean = self._biased_mean / (1 - self.decay ** self.n)
