from chainer import functions as F, Chain, as_variable, report

from experiments.util.bandify import RankingBandify
from experiments.util.propensity_estimator import PresentationBiasPropensityEstimator


class Counterfactual(Chain):
    def __init__(self, policy_to_optimize, logging_policy, click_model,
                 eta=1.0):
        """
        A counterfactual ranking objective

        :param policy_to_optimize: The ranker policy we wish to optimize
        :type policy_to_optimize: chainercb.policy.Policy

        :param logging_policy: The logging policy used to generate rankings
        :type logging_policy: chainercb.policy.Policy

        :param behavior: The behavior
        :type behavior: chainerltr.clickmodels.behavior.UserBehavior

        :param eta: Parameter controlling assumed presentation bias
        :type eta: float
        """
        super().__init__(policy_to_optimize=policy_to_optimize)
        with super().init_scope():
            self.bandify = RankingBandify(logging_policy, click_model)
            self.propensity_estimator = PresentationBiasPropensityEstimator(eta)

    def __call__(self, x, y, nr_docs):
        # Turn supervised into a partial feedback problem using our logging
        # policy
        x, y, nr_docs = as_variable(x), as_variable(y), as_variable(nr_docs)
        x, a, log_p, r = self.bandify(x, y, nr_docs)
        log_p = self.propensity_estimator(log_p)
        log_p_to_optimize = self.policy_to_optimize.log_propensity(x, a)
        report({"avg_p": F.mean(F.exp(log_p[:, 0]))}, self)
        return self.loss(x, a, log_p, log_p_to_optimize, r)

    def loss(self, x, a, log_p, log_p_to_optimize, r):
        """
        Computes a counterfactual loss

        :param x: The observations
        :type x: chainer.Variable

        :param a: The actions
        :type a: chainer.Variable

        :param log_p: The log propensity scores
        :type log_p: chainer.Variable

        :param log_p_to_optimize: Policy propensity scores to optimize
        :type log_p_to_optimize: chainer.Variable

        :param r: Rewards
        :type r: chainer.Variable

        :return: The loss
        :rtype: chainer.Variable
        """
        pass
