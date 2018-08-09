"""SMC sampling methods."""

import logging

import numpy as np
import matplotlib.pyplot as plt

from elfi.methods.utils import (GMDistribution, weighted_var)

logger = logging.getLogger(__name__)



def smc(n_samples, prior, iterations, params0, target, seed=0):
    """Sample the target with a Sequential Monte Carlo using Gaussian proposals.

    Parameters
    ----------
    n_samples : int
        The number of requested samples.
    prior : function
        The prior distribution of the model.
    iterations :
        Maximum number of iterations for the SMC algorithm.
    params0 : np.array
        Initial values for each sampled parameter.
    target : function
        The target log density to sample (possibly unnormalized).
    seed : int, optional
        Seed for pseudo-random number generator.

    Returns
    -------
    samples : np.array

    """

    random_state = np.random.RandomState(seed)
    samples = prior.rvs(size=n_samples, random_state=random_state)# how to sample from prior?
    w = np.ones(n_samples)
    cov = 2 * np.diag(weighted_var(samples, w))
    for i in range(1,iterations):
        samples_old = samples
        samples = GMDistribution.rvs(means=samples_old,cov=cov,size=n_samples)
        q_logpdf = GMDistribution.logpdf(x=samples,means=list(samples_old),cov=cov)
        p_logpdf = target(samples)
        w = np.exp(p_logpdf - np.max(p_logpdf) - q_logpdf)
        cov = 2 * np.diag(weighted_var(samples, w/np.sum(w)))
        ind = np.random.choice(np.arange(n_samples), n_samples, replace=True, p = w/np.sum(w))
        samples = samples[ind,]
        w = np.ones(n_samples)

        if np.count_nonzero(w) == 0:
            raise RuntimeError("All sample weights are zero. If you are using a prior "
            "with a bounded support, this may be caused by specifying "
            "a too small sample size.")

    return samples
