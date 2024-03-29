import numpy as np
# import scipy.stats
# import scipy.special
from scipy.special import (
    gammaln,
    betainc,
)

try:
    import xarray as xr
except ImportError:
    xr = None

def _broadcast(*args):
    # broadcasting
    if xr and all([isinstance(i, xr.DataArray) for i in args]):
        return xr.align(*args)
    else:
        return np.broadcast_arrays(*args)

class NegativeBinomial:
    r"""
    Negative binomial distribution.
    This class supports re-parameterising, sampling and calculation of
    probabilities of negative binomial distributed data.
    """

    mean: np.ndarray
    # variance: np.ndarray
    # p: np.ndarray
    r: np.ndarray

    @classmethod
    def mme(cls, data, axis=0):
        r"""
        Fit a Negative Binomial distribution to `data`.
        Uses the closed-form Method-of-Moments to estimate the dispersion.

        :param data: The data
        :param axis: Axis along which the distributions are provided
        :return: `NegativeBinomial` object

        """
        mean = np.mean(data, axis=axis)
        variance = np.mean(np.square(data - mean), axis=axis)

        return cls(mean=mean, variance=variance)

    def __init__(self, r=None, variance=None, p=None, mean=None):
        if r is not None:
            if variance is not None:
                raise ValueError("Must pass either shape 'r' or variance, but not both")

            if p is not None:
                if mean is not None:
                    raise ValueError("Must pass either probs or means, but not both")

                mean = p * r / (1 - p)
                # variance = mean / (1 - p)

            elif mean is not None:
                if p is not None:
                    raise ValueError("Must pass either probs or means, but not both")

                # p = mean / (r + mean)
                # variance = mean + (np.square(mean) / r)

            else:
                raise ValueError("Must pass probs or means")

        elif variance is not None:
            if r is not None:
                raise ValueError("Must pass either shape 'r' or variance, but not both")

            if p is not None:
                if mean is not None:
                    raise ValueError("Must pass either probs or means, but not both")

                mean = variance * (1 - p)
                r = mean * mean / (variance - mean)

            elif mean is not None:
                if p is not None:
                    raise ValueError("Must pass either probs or means, but not both")

                # p = 1 - (mean / variance)
                r = mean * mean / (variance - mean)
            else:
                raise ValueError("Must pass probs or means")
        else:
            raise ValueError("Must pass shape 'r' or variance")

        self.mean = mean
        # self.variance = variance
        # self.p = p
        self.r = r

    @property
    def variance(self):
        return self.mean + (np.square(self.mean) / self.r)

    @property
    def p(self):
        return self.mean.astype("float64") / (self.r + self.mean).astype("float64")

    @property
    def log_p(self):
        return np.log(self.mean.astype("float64")) - np.log((self.r + self.mean).astype("float64"))

    @property
    def log_1p(self):
        return np.log(self.r.astype("float64")) - np.log((self.r + self.mean).astype("float64"))

    def sample(self, size=None):
        """
        Sample from all distributions data of size `size`.
        :param size: The size
        :return: numpy array containing sampled data
        """
        # numpy uses an alternative parametrization
        # see also https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
        random_data = np.random.negative_binomial(
            n=self.r,
            p=1 - self.p,
            size=size
        )
        return random_data

    def prob(self, X):
        """
        Calculate the probability of each value in `X` given this distribution
        :param X: The data
        :return: numpy array of probabilitites
        """
        # p = self.p
        # r = self.r
        # return scipy.stats.nbinom(n=r, p=1 - p).pmf(X)
        # return binom(X + r - 1, X) * np.power(p, X) * np.power(1 - p, r)
        return np.exp(self.log_prob(X))

    def log_prob(self, X):
        """
        Calculate the log-probability of each value in `X` given this distribution
        :param X: The data
        :return: numpy array of log-probabilitites
        """
        from scipy.special import (
            gammaln,
            betainc,
        )
        mu = self.mean
        r = self.r

        # min_p = np.nextafter(0, 1, dtype=self.p.dtype)
        # max_p = np.nextafter(1, 0, dtype=self.p.dtype)
        # log_p = np.fmin(log_p, max_p)
        # log_p = np.fmax(log_p, min_p)

        # broadcasting
        if xr and isinstance(mu, xr.DataArray) and isinstance(r, xr.DataArray) and isinstance(X, xr.DataArray):
            mu, r, X = xr.align(mu, r, X)
        else:
            mu, r, X = np.broadcast_arrays(mu, r, X)

        div = np.log(r + mu)
        log_p = np.log(mu) - div
        log_1p = np.log(r) - div

        # return scipy.stats.nbinom(n=r, p=1 - p).logpmf(X)
        coeff = gammaln(r + X) - gammaln(X + 1) - gammaln(r)
        # return coeff + r * np.log(1 - p) + X * np.log(p)
        return coeff + r * log_1p + X * log_p

    def cdf(self, X):
        r = self.r
        log_1p = self.log_1p

        # broadcasting
        log_1p, r, X = _broadcast(log_1p, r, X)

        if xr:
            return xr.apply_ufunc(scipy.stats.nbinom.cdf, X, r, np.exp(log_1p), dask="parallelized")
        else:
            return scipy.stats.nbinom.cdf(X, r, np.exp(log_1p))

    def log_cdf(self, X):
        r = self.r
        log_1p = self.log_1p

        # broadcasting
        log_1p, r, X = _broadcast(log_1p, r, X)

        if xr:
            return xr.apply_ufunc(scipy.stats.nbinom.logcdf, X, r, np.exp(log_1p), dask="parallelized")
        else:
            return scipy.stats.nbinom.logcdf(X, r, np.exp(log_1p))

    def pval(self, X, alternative=None):
        """
        Compute p-value for given set of counts
        :param X: The data
        :param alternative: If `alternative` is set to "less" or "greater" it will return
            the corresponding alternative p-value instead of the default two-sided p-value
        """
        cdf = self.cdf(X)
        # cdf = np.exp(self.log_cdf(X))
        density = self.prob(X)

        pval = np.fmin(0.5, np.fmin(cdf, 1 - cdf + density)) * 2

        p_less = cdf
        if alternative == "less":
            return p_less

        p_greater = 1 - cdf + density
        if alternative == "greater":
            return p_greater

        return np.fmin(0.5, np.fmin(p_less, p_greater)) * 2
        # return betainc(r, 1. + X, np.exp(log_1p))


class Normal:
    r"""
    Normal distribution.
    """

    mean: np.ndarray
    sd: np.ndarray

    def __init__(self, mean, sd):
        self.sd = sd
        self.mean = mean

    def sample(self, size=None):
        """
        Sample from all distributions data of size `size`.
        :param size: The size
        :return: numpy array containing sampled data

        """
        random_data = np.random.normal(
            loc=self.mean,
            scale=self.sd,
            size=size
        )
        return random_data


class Beta:
    r"""
    Beta distribution.
    """

    p: np.ndarray
    q: np.ndarray

    def __init__(self, mean, samplesize):
        self.p = mean * samplesize
        self.q = (1 - mean) * samplesize

    def sample(self, size=None):
        """
        Sample from all distributions data of size `size`.
        :param size: The size
        :return: numpy array containing sampled data

        """
        random_data = np.random.beta(
            a=self.p,
            b=self.q,
            size=size
        )
        return random_data
