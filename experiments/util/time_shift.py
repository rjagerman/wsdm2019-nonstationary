from math import floor

from scipy.special import betainc


class DictShift:
    def __init__(self, dict, key, shift_fn):
        self.shift_fn = shift_fn
        self.dict = dict
        self.key = key

    def __call__(self, t):
        v = self.shift_fn(t)
        self.dict[self.key] = v


class ObjectShift(DictShift):
    def __init__(self, obj, key, shift_fn):
        super().__init__(obj.__dict__, key, shift_fn)


class TransformShift:
    def __init__(self, minimum, maximum, shift_fn):
        self.minimum = minimum
        self.maximum = maximum
        self.shift_fn = shift_fn

    def __call__(self, t):
        v = self.shift_fn(t)
        return self.minimum + v * (self.maximum - self.minimum)


class DurationShift:
    def __init__(self, start, end, shift_fn):
        self.start = start
        self.end = end
        self.shift_fn = shift_fn

    def __call__(self, t):
        if self.start <= t <= self.end:
            t = (t - self.start) / (self.end - self.start)
            return self.shift_fn(t)
        else:
            return None


class CompositeShift:
    def __init__(self, shift_fns, default):
        self.shift_fns = shift_fns
        self.default = default
        self._last_value = self.default

    def __call__(self, t):
        for shift_fn in self.shift_fns:
            v = shift_fn(t)
            if v is not None:
                self._last_value = v
        return self._last_value

    def append(self, shift_fn):
        self.shift_fns.append(shift_fn)


class AbruptShift:
    def __call__(self, t):
        return floor(t)


class LinearShift:
    def __call__(self, t):
        return t


class IncBetaShift:
    def __init__(self, in_curve, out_curve = None):
        self.in_curve = in_curve
        self.out_curve = out_curve
        if self.out_curve is None:
            self.out_curve = in_curve

    def __call__(self, t):
        return betainc(self.in_curve, self.out_curve, t)
