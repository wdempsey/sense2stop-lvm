# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:29:26 2020

@author: Walter Dempsey & Jamie Yap
"""

#%%

###############################################################################
# Build a RJMCMC class
###############################################################################

from pymc import Stochastic, Deterministic, Node, StepMethod
from numpy import ma, random, where
from numpy.random import random
from copy import deepcopy

class smartdumbRJ(StepMethod):
    """
    S = smartdumbRJ(self, stochs, indicator, p, rp, g, q, rq, inv_q, Jacobian, **kwargs)
    smartdumbRJcan control single indicatored-array-valued stochs. The indicator
    indicates which stochs (events) are currently 'in the model;' if
    stoch.value.indicator[index] = True, that index is currently being excluded.
    indicatored-array-valued stochs and their children should understand how to
    cope with indicatored arrays when evaluating their logpabilities.
    The prior for the indicatored-array-valued stoch may depend explicitly on the
    indicator.
    The dtrm arguments are, in notation similar to that of Waagepetersen et al.,
        def p(indicator):
            Returns the probability of jumping  to 
        def smartbirth(indicator):
            Draws a value for the auxiliary RV's u given indicator.value (proposed),
            indicator.last_value (current), and the value of the stochs.
        def smartdeath(indicator):
    """
    def __init__(self, stochs, indicator, p, rp, g, q, rq, inv_q, Jacobian):

        StepMethod.__init__(self, nodes = stochs)

        self.g = g
        self.q = q
        self.rq = rq
        self.p = p
        self.rp = rp
        self.inv_q = inv_q
        self.Jacobian = Jacobian

        self.stoch_dict = {}
        for stoch in stochs:
            self.stoch_dict[stoch.__name__] = stoch

        self.indicator = indicator


    def propose(self):
        """
        Sample a new indicator and value for the stoch.
        """
        self.rp(self.indicator)
        self._u = self.rq(self.indicator)
        self.g(self.indicator, self._u, **self.stoch_dict)



    def step(self):
        # logpability and loglike for stoch's current value:
        logp = sum([stoch.logp for stoch in self.stochs]) + self.indicator.logp
        loglike = self.loglike

        # Sample a candidate value for the value and indicator of the stoch.
        self.propose()

        # logpability and loglike for stoch's proposed value:
        logp_p = sum([stoch.logp for stoch in self.stochs]) + self.indicator.logp

        # Skip the rest if a bad value is proposed
        if logp_p == -Inf:
            for stoch in self.stochs: stoch.revert()
            return

        loglike_p = self.loglike

        # test:
        test_val =  logp_p + loglike_p - logp - loglike
        test_val += self.inv_q(self.indicator)
        test_val += self.q(self.indicator,self._u)

        if self.Jacobian is not None:
            test_val += self.Jacobian(self.indicator,self._u,**self.stoch_dict)

        if log(random()) > test_val:
            for stoch in self.stochs:
                stoch.revert


    def tune(self):
        pass


