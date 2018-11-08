import numpy as np
from chainer import as_variable, functions as F
from chainer.backends import cuda
from chainercb.policy import Policy
from chainerltr.functions import sample_without_replacement, \
    inverse_select_items_per_row, select_items_per_row, logcumsumexp
from scipy.special import gamma, gammaln


class PlackettLuce(Policy):
    def __init__(self, predictor, tau=1.0, k=0, threshold=None):
        """
        A policy that repurposes the output of a prediction function via the
        Plackett-Luce distribution to sample rankings.

        :param predictor: The predictor function
        :type predictor: chainer.Link

        :param tau: The temperature parameter dictating the smoothness
        :type tau: float

        :param k: The cut-off for the rankings
        :type k: int
        """
        super().__init__(predictor=predictor)
        self.tau = tau
        self.k = k
        self.threshold = threshold

    def draw(self, x):
        log_p = F.log_softmax(self._predict(x))
        action = as_variable(sample_without_replacement(log_p))
        return self._cut(action)

    def max(self, x):
        pred = self._predict(x)
        xp = cuda.get_array_module(pred)

        # We randomly permute to break ties by random shuffling
        idx = as_variable(xp.random.uniform(0.0, 1.0, pred.shape).argsort())
        pred = select_items_per_row(pred, idx)

        # Perform argsort
        action = as_variable(xp.fliplr(xp.argsort(pred.data, axis=1)))

        # Revert original permutation
        action = select_items_per_row(idx, action)

        return self._cut(action)

    def uniform(self, x):
        pred = self._predict(x)
        xp = cuda.get_array_module(pred)
        log_p = F.log_softmax(
            as_variable(xp.ones(pred.shape, dtype=pred.dtype)))
        action = as_variable(sample_without_replacement(log_p))
        return self._cut(action)

    def nr_actions(self, x):
        xp = cuda.get_array_module(x)
        pred = self._predict(x)
        nr_docs = pred.shape[1]
        nr_actions = gamma(nr_docs + 1)
        if self.k > 0 and nr_actions < np.inf:
            nr_actions /= gamma(nr_docs + 1 - self.k)
        return as_variable(xp.ones(pred.shape[0]) * nr_actions)

    def log_nr_actions(self, x):
        pred = self._predict(x)
        nr_docs = pred.shape[1]
        log_nr_actions = gammaln(nr_docs + 1)
        if self.k > 0:
            log_nr_actions -= gammaln(nr_docs + 1 - self.k)
        xp = cuda.get_array_module(pred)
        return as_variable(xp.ones(pred.shape[0]) * log_nr_actions)

    def propensity(self, x, action):
        return F.exp(self.log_propensity(x, action))

    def log_propensity(self, x, action):
        xp = cuda.get_array_module(action)
        pred = self._predict(x)

        final_action = action
        if self.k > 0 and action.shape[1] < pred.shape[1]:
            all_actions = F.broadcast_to(xp.arange(0, pred.shape[1],
                                                   dtype=action.data.dtype),
                                         pred.shape)
            inv_items = inverse_select_items_per_row(all_actions, action)
            items = select_items_per_row(all_actions, action)
            final_action = F.concat((items, inv_items), axis=1)

        pred = select_items_per_row(pred, final_action)

        results = pred - logcumsumexp(pred)
        if self.k > 0:
            results = results[:, :self.k]
        return results

    def log_propensity_independent(self, x, action):
        xp = cuda.get_array_module(action)
        pred = self._predict(x)

        final_action = action
        if self.k > 0 and action.shape[1] < pred.shape[1]:
            all_actions = F.broadcast_to(xp.arange(0, pred.shape[1],
                                                   dtype=action.data.dtype),
                                         pred.shape)
            inv_items = inverse_select_items_per_row(all_actions, action)
            items = select_items_per_row(all_actions, action)
            final_action = F.concat((items, inv_items), axis=1)

        pred = select_items_per_row(pred, final_action)

        results = F.log_softmax(pred)
        if self.k > 0:
            results = results[:, :self.k]
        return results

    def _predict(self, x):
        """
        Generates a prediction and rescales it with the temperature parameter

        :param x: The context vectors
        :type x: chainer.Variable

        :return: The predictions made by the prediction function
        :type: chainer.Variable
        """
        output = self.predictor(x) / self.tau

        # Optional per-batch thresholding to prevent numerical instability
        if self.threshold is not None and F.max(
                F.absolute(output)).data > self.threshold:
            output = output / F.max(output).data * self.threshold

        return output

    def _cut(self, action):
        """
        Cuts the action of at rank k

        :param action: The action
        :type action: chainer.Variable

        :return: The action cut-off at rank k
        :rtype: chainer.Variable
        """
        if 0 < self.k < action.shape[1]:
            action_before = action[:, :self.k].data
            action_after = action[:, self.k:].data
            xp = cuda.get_array_module(action_after)
            xp.random.shuffle(action_after)
            action = F.concat((action_before, action_after))
        return action
