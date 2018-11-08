from chainer import as_variable
from chainercb.bandify import Bandify


class RankingBandify(Bandify):
    def __init__(self, acting_policy, click_model, rng=None):
        """

        :param acting_policy:
        :type acting_policy: chainercb.policy.Policy

        :param click_model:
        :type click_model: chainerltr.clickmodels.clickmodel.ClickModel

        :param rng: Function to get a random number generator
        :type rng: () => numpy.random.RandomState
        """
        super().__init__(acting_policy)
        self.click_model = click_model
        self.nr_docs = None
        self.rng = rng

    def __call__(self, *args):
        if len(args) != 3:
            raise RuntimeError(
                f"expecting 3 arguments for RankingBandify, got {len(args)}")
        observations, labels, nr_docs = args
        self.nr_docs = nr_docs
        return super().__call__(observations, labels)

    def reward(self, actions, labels, dtype):
        click_vector = self.click_model(actions, labels, self.nr_docs)
        click_vector = as_variable(click_vector.data.astype(dtype))
        return click_vector
