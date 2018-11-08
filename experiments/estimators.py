import numpy as np
from chainer import functions as F, as_variable
from chainer.backends import cuda
from chainercb.util.ridge import RidgeRegression
from scipy.optimize import minimize_scalar
from collections import deque


class Estimator:
    def __init__(self, policy_to_evaluate):
        self.policy_to_evaluate = policy_to_evaluate

    def __call__(self, x, a, p, r):
        return self.reward(x, a, p, r)

    def reward(self, x, a, p, r):
        """
        Computes a point-wise per-sample reward estimate of given batch x

        :param x: The batch of feature vectors
        :type x: chainer.Variable

        :param a: The actions that were taken by the logging policy
        :type a: chainer.Variable

        :param p: The propensity score of the logging policy for each action
        :type p: chainer.Variable

        :param r: The rewards for the logging policy
        :type r: chainer.Variable

        :return: The per-sample reward estimate of policy_to_evaluate
        :rtype: chainer.Variable
        """
        raise NotImplementedError


class IPS(Estimator):
    """
    Vanilla IPS estimator
    """
    def reward(self, x, a, p, r):
        return self.policy_to_evaluate.propensity(x, a) / p * r


class DMADF(Estimator):
    """
    Direct-Method estimator for ADF-type actions
    """
    def __init__(self, policy_to_evaluate, d):
        super().__init__(policy_to_evaluate)
        self.regressor = RidgeRegression(d)

    def reward(self, x, a, p, r):
        xp = cuda.get_array_module(x)

        if r is not None:
            # Select the actions that the policy to evaluate would have chosen
            actions = self.policy_to_evaluate.draw(x).data
        else:
            # Select provided actions when no logged reward is provided
            actions = a.data

        # Compute a reward estimate via the regressor
        output = self.regressor.predict(x[xp.arange(x.shape[0]), actions, :])

        # Update regressor with new reward information (if available)
        if r is not None:
            self.regressor.update(x[xp.arange(x.shape[0]), a.data, :], r)

        # Return output
        return output


class DRADF(Estimator):
    """
    Doubly-Robust estimator for ADF-type actions
    """
    def __init__(self, policy_to_evaluate, d):
        super().__init__(policy_to_evaluate)
        self.dm = DMADF(policy_to_evaluate, d)
        self.ips = IPS(policy_to_evaluate)

    def reward(self, x, a, p, r):
        r_hat_log = self.dm(x, a, p, None)
        r_hat = self.dm(x, a, p, r)
        return self.ips(x, a, p, r - r_hat_log) + r_hat


class Aggregator:
    def __init__(self, estimator):
        self.estimator = estimator

    def __call__(self, x, a, p, r):
        r_hat = self.estimator(x, a, p, r)
        return self.add(r_hat)

    def add(self, r_hat):
        """
        Adds a batch of estimated rewards to the aggregator, and then returns
        the current estimated aggregate reward

        :param r_hat: The reward estimation
        :type r_hat: chainer.Variable

        :return: The estimated overal reward of the policy
        :rtype: float
        """
        raise NotImplementedError


class AverageAggregator(Aggregator):
    def __init__(self, estimator):
        super().__init__(estimator)
        self.sum = 0.0
        self.n = 0

    def reset(self):
        self.sum = 0.0
        self.n = 0

    def add(self, r_hat):
        self.sum += F.sum(r_hat).data
        self.n += r_hat.shape[0]
        return self.sum / self.n

    @property
    def mean(self):
        return self.sum / self.n


class ExponentialAggregator(Aggregator):
    def __init__(self, estimator, alpha=0.99):
        super().__init__(estimator)
        self.biased_mean = 0.0

        self.n = 0
        self.alpha = alpha
        self.bias_correction_factor = 1.0
        self.mean = 0.0
        self.var = 0.0
        self.should_print = False

    def add(self, r_hat):
        xp = cuda.get_array_module(r_hat)
        weights = self.alpha ** (xp.arange(r_hat.shape[0], 0, -1.0) - 1.0)
        summation = (1 - self.alpha) * F.sum(as_variable(weights) * r_hat,
                                             axis=0)

        # Compute exponential moving variance
        batch = (1 - self.alpha) * F.cumsum(as_variable(weights) * r_hat, axis=0) + (self.alpha ** r_hat.shape[0]) * self.biased_mean
        batch = batch.data / (1 - self.alpha ** (self.n + 1.0 + xp.arange(0.0, r_hat.shape[0], 1.0)))
        batch_s = xp.roll(batch, 1, axis=0)
        batch_s[0] = self.mean
        diff = r_hat - batch_s
        diff = self.alpha * (diff ** 2)
        var = (1 - self.alpha) * F.sum(as_variable(weights) * diff, axis=0)
        self.var = (self.alpha ** r_hat.shape[0]) * self.var + var.data

        # Compute exponential moving average
        self.biased_mean = (self.alpha ** r_hat.shape[0]) * self.biased_mean + summation.data
        self.n += r_hat.shape[0]
        self.bias_correction_factor *= self.alpha ** r_hat.shape[0]
        self.mean = self.biased_mean / (1 - self.bias_correction_factor)

        return self.mean


class WindowAggregator(Aggregator):
    def __init__(self, estimator, tau=100):
        super().__init__(estimator)
        self.window = deque()
        self.tau = tau
        self.should_print = False

    def add(self, r_hat):
        xp = cuda.get_array_module(r_hat)
        self.window.extend(r_hat.data)
        while len(self.window) > self.tau:
            self.window.popleft()
        return xp.mean(self.window)

    @property
    def mean(self):
        return np.mean(self.window)


class LipschitzEstimator(Aggregator):
    def __init__(self, estimator, avg, change_detection_window=1000):
        super().__init__(estimator)
        self.window = deque(maxlen=change_detection_window)
        self.change_detection_window = change_detection_window
        self.current_avg = avg()
        # self.previous_avg = avg()
        self.smoothing = deque(maxlen=5)
        self.vc = 0.0
        self.vp = 0.0

    def add(self, values):
        # self.window.extend(values.data)
        cv = self.current_avg.add(values.data)
        self.window.append(cv)
        return self.estimate()

    def estimate(self):
        k = 0.0
        if len(self.window) == self.change_detection_window:
            self.vc = self.current_avg.mean
            self.vp = self.window[0]
            k = (self.vc - self.vp) / len(self.window)
        self.smoothing.append(k)
        return np.mean(list(self.smoothing))


class AdaptiveExponentialAggregator(Aggregator):
    def __init__(self, estimator, alpha=0.999, factor=0.002, should_print=False,
                 adapt_window=1000):
        super().__init__(estimator)
        self.initial_alpha = alpha
        self.alpha = alpha
        self.factor = factor
        self.k = 0.0
        self.n = 0
        self.should_print = should_print
        self.adaptive_estimator = ExponentialAggregator(None, alpha)
        self.lipschitz_estimator = LipschitzEstimator(estimator, lambda: ExponentialAggregator(None, alpha), adapt_window)
        self.adapt_last = 0
        self.adapt_every = adapt_window

    def add(self, r_hat):

        # Update estimator statistics and get current estimate
        self.n += r_hat.shape[0]
        self.k = self.lipschitz_estimator.add(r_hat)

        # Find optimal alpha
        if self.k == 0:
            self.alpha = self.initial_alpha
        else:
            # We can compute a closed-form of alpha
            sqk = np.sqrt(self.factor * np.abs(self.k))
            sq2 = np.sqrt(2)
            self.alpha = np.clip(-(sqk - sq2) / (sqk + sq2), 1e-10, 1.0 - 1e-10)

        # Set alpha
        self.adaptive_estimator.alpha = self.alpha

        if self.should_print:
            print("=" * 100)
            print(f"n: {self.n}")
            print(f"vc: {self.lipschitz_estimator.vc}")
            print(f"vp: {self.lipschitz_estimator.vp}")
            print(f"alpha: {self.adaptive_estimator.alpha}")
            print(f"k: {self.k}")
            print(f"est: {self.adaptive_estimator.mean}")

        # Return result
        result = self.adaptive_estimator.add(r_hat)
        return result


class AdaptiveWindowAggregator(Aggregator):
    def __init__(self, estimator, tau=0.999, factor=0.002, should_print=False,
                 adapt_window=1000):
        super().__init__(estimator)
        self.intial_tau = tau
        self.tau = tau
        self.factor = factor
        self.k = 0.0
        self.n = 0
        self.should_print = should_print
        self.adaptive_estimator = WindowAggregator(None, tau)
        self.lipschitz_estimator = LipschitzEstimator(estimator, lambda: WindowAggregator(None, tau), adapt_window)

    def add(self, r_hat):

        # Update estimator statistics and get current estimate
        self.n += r_hat.shape[0]
        self.k = self.lipschitz_estimator.add(r_hat)

        # Find optimal tau
        if self.k == 0:
            self.tau = self.intial_tau
        else:
            # We can compute a closed-form of tau
            tau_est = np.sqrt(2 / np.abs(self.factor * self.k))
            self.tau = np.clip(tau_est, 500, 500000)

        # Set tau
        self.adaptive_estimator.tau = self.tau

        if self.should_print:
            print("=" * 100)
            print(f"n: {self.n}")
            print(f"vc: {self.lipschitz_estimator.vc}")
            print(f"vp: {self.lipschitz_estimator.vp}")
            print(f"tau: {self.adaptive_estimator.tau}")
            print(f"k: {self.k}")
            print(f"est: {self.adaptive_estimator.mean}")

        # Return result
        result = self.adaptive_estimator.add(r_hat)
        return result
