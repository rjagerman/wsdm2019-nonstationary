from chainer import Chain, as_variable
from chainer.backends import cuda
from chainer.training import extension


class RewardShiftExtension(extension.Extension):
    def __init__(self, shifter):
        self.shifter = shifter

    def __call__(self, trainer):
        self.shifter.set_time(trainer.updater.iteration)


class SeasonalSimulation:
    def __init__(self, delay, attack, sustain, release, cycle=True):
        """
        Simulates simple seasonality spike (a value going from 0.0 to 1.0 and
        back down). The seasonal trend can happen only once or cycle.

        :param delay: The delay where rewards are the original
        :type delay: int

        :param attack: Time it takes to build up to the modified rewards
        :type attack: int

        :param sustain: Time for the modified rewards to be active
        :type sustain: int

        :param release: Time it takes to fall back to the original rewards
        :type sustain: int

        :param cycle: Whether to cycle or not
        :type cycle: bool
        """
        self.delay = delay
        self.attack = attack
        self.sustain = sustain
        self.release = release
        self.cycle = cycle

    @property
    def cycle_time(self):
        """
        :return: The total time for a single cycle
        :rtype: int
        """
        return self.delay + self.attack + self.sustain + self.release

    def __call__(self, t):
        if self.cycle:
            t = t % self.cycle_time
        out = 0.0
        start_ramp = self.delay
        start_sustain = self.delay + self.attack
        start_release = self.delay + self.attack + self.sustain
        end = self.cycle_time
        if start_ramp < t < start_sustain:
            out = float(t - start_ramp) / float(self.attack)
        if start_sustain <= t <= start_release:
            out = 1.0
        if start_release < t < end:
            out = 1.0 - (float(t - start_release) / float(self.release))
        return out


class RewardShifter(Chain):
    def __init__(self, modifier, chain, mapping):
        super().__init__(chain=chain)
        self.modifier = modifier
        self.mapping = mapping
        self._t = 0

    def set_time(self, t):
        self._t = t

    def __call__(self, xs, ys, nr_docs):
        """
        Shifts the labels of given batch of relevance labels

        :param xs: The batch of observations
        :type xs: chainer.Variable

        :param ys: The relevance labels that will be shifted
        :type ys: chainer.Variable

        :param nr_docs: The batch of nr of doc counters
        :type nr_docs: chainer.Variable

        :return: The shifted relevance labels
        :rtype: chainer.Variable
        """
        xp = cuda.get_array_module(ys)
        ys = as_variable(ys)
        y = ys.data
        p = xp.ones_like(y) * self.modifier(self._t)
        r = (p >= xp.random.uniform(size=y.shape)) * 1.0
        y = self.mapping(y) * r + y * (1.0 - r)
        return self.chain(as_variable(xs), as_variable(y), as_variable(nr_docs))


def flip_2_4_mapping(y):
    xp = cuda.get_array_module(y)
    y_output = xp.copy(y)
    y_output[y == 4.0] = 0.0
    y_output[y == 3.0] = 0.0
    y_output[y == 2.0] = 0.0
    y_output[y == 1.0] = 0.0
    y_output[y == 0.0] = 4.0
    return y_output
