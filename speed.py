import math
import numpy as np


class SpeedFunc:
    def __init__(self, delta_t=0.01):
        self.delta_t = delta_t

    def __call__(self, clip_index):
        raise NotImplementedError


class StaticSpeed(SpeedFunc):
    def __init__(self):
        super().__init__()

    def __call__(self, clip_index):
        return 1


class SinSpeed(SpeedFunc):
    def __init__(self, cycle_t: float, offset_s: float, scale_s: tuple, delta_t=0.01):
        super().__init__(delta_t)
        self.cycle_t = cycle_t
        self.offset_s = offset_s
        self.scale_s = scale_s

    def __call__(self, clip_index):
        cycle = self.cycle_t / self.delta_t
        speed = np.sin((2 * math.pi) * clip_index / cycle)
        if isinstance(speed, np.ndarray):
            mask = speed >= 0
            speed[mask] = speed[mask] * self.scale_s[0] + self.offset_s
            speed[~mask] = speed[~mask] * self.scale_s[1] + self.offset_s
        else:
            speed = speed * self.scale_s[0 if speed >= 0 else 1] + self.offset_s
        return speed


class SquareWaveSpeed(SpeedFunc):
    def __init__(self, cycle_t: tuple, scale_s: tuple, delta_t=0.01):
        super().__init__(delta_t)
        self.cycle_t = cycle_t
        self.cycle_cum = np.cumsum(cycle_t)
        self.scale_s = np.array(scale_s)

    def __call__(self, clip_index):
        t = (clip_index * self.delta_t) % self.cycle_cum[-1]
        idx = np.searchsorted(self.cycle_cum, t)
        return self.scale_s[idx]


class SawtoothSpeed(SpeedFunc):
    def __init__(self, cycle_t: float, scale_s: tuple, delta_t=0.01):
        super().__init__(delta_t)
        assert len(scale_s) == 2
        self.cycle_t = cycle_t
        self.scale_s = scale_s

    def __call__(self, clip_index):
        t = (clip_index * self.delta_t) % self.cycle_t
        weight = t / self.cycle_t
        return weight * self.scale_s[1] + (1 - weight) * self.scale_s[0]


class RandomSpeed(SpeedFunc):
    def __init__(self, scale_s: tuple, seed=114514, delta_t=3):
        super().__init__(delta_t)
        assert len(scale_s) == 2
        self.scale_s = scale_s
        np.random.seed(seed)

    def __call__(self, clip_index):
        return np.random.rand(1)[0] * (self.scale_s[1] - self.scale_s[0]) + self.scale_s[0]


class ExpSpeed(SpeedFunc):
    def __init__(self, scale_s: float = 0.05, delta_t=0.01):
        super().__init__(delta_t)
        self.scale_s = scale_s

    def __call__(self, clip_index):
        t = clip_index * self.delta_t
        return math.exp(self.scale_s * t)


class LogSpeed(SpeedFunc):
    def __init__(self, scale_s: float = 0.5, delta_t=0.01):
        super().__init__(delta_t)
        self.scale_s = scale_s

    def __call__(self, clip_index):
        t = clip_index * self.delta_t
        return math.log(self.scale_s * t + 1)


class SigmoidSpeed(SpeedFunc):
    def __init__(self, offset_t: float = 20, scale_s: tuple = (1, 3), delta_t=0.01):
        super().__init__(delta_t)
        self.offset_t = offset_t
        self.scale_s = scale_s

    def __call__(self, clip_index):
        t = clip_index * self.delta_t - self.offset_t
        sp = 1 / (1 + math.exp(-t))
        return (sp + 1) / 2 * (self.scale_s[1] - self.scale_s[0]) + self.scale_s[0]


class XSinSpeed(SpeedFunc):
    def __init__(self, cycle_t: float, scale_t: float, delta_t=0.01):
        super().__init__(delta_t)
        self.cycle_t = cycle_t
        self.scale_t = scale_t

    def __call__(self, clip_index):
        t = clip_index * self.delta_t
        cycle = self.cycle_t / self.delta_t
        speed = abs(np.sin((2 * math.pi) * clip_index / cycle))
        return speed * (t * self.scale_t)


class PowerSpeed(SpeedFunc):
    def __init__(self, power: float, scale_t: float, delta_t=0.01):
        super().__init__(delta_t)
        self.power = power
        self.scale_t = scale_t

    def __call__(self, clip_index):
        t = clip_index * self.delta_t
        return (t * self.scale_t) ** self.power


class TanSpeed(SpeedFunc):
    def __init__(self, scale_t: float = 55, delta_t=0.001):
        super().__init__(delta_t)
        self.scale_t = scale_t

    def __call__(self, clip_index):
        t = clip_index * self.delta_t
        return math.tan(t / self.scale_t) + 1


class ArcsinhSpeed(SpeedFunc):
    def __init__(self, delta_t=0.01):
        super().__init__(delta_t)

    def __call__(self, clip_index):
        t = clip_index * self.delta_t
        return math.asinh(t)
