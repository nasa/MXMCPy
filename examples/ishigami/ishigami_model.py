from math import sin

import numpy as np


class IshigamiModel:
    """
    Defines Ishigami model with three free params (desc...). The
    quantity of interest that is returned by the evaluate() function is the
    ........................
    """

    def __init__(self, a=7., b=.1, state0=None, time_step=None, cost=None):

        self._a = a
        self._b = b

    def simulate(self, inputs):
        """
        Run Ishigami function for inputs x1...x3 drawn from
        uniform distributions in the range -pi to pi.
        """
        if inputs.size != 3:
            raise ValueError("Inputs to simulate should have 3 elements.")

        x1, x2, x3 = inputs

        return self._ishigami_func(x1, x2, x3)

    def evaluate(self, inputs):
        """
        Returns the max displacement over the course of the simulation.
        MLMCPy convention is that evaluated takes in an array and returns an
        array (even for 1D examples like this one).
        """
        return self.simulate(inputs)

    def _ishigami_func(self, x1, x2, x3):
        """
        Return result of Ishigami function for given xi, i=1...3
        """

        # Compute each term.
        term1 = sin(x1)
        term2 = self._a * sin(x2) ** 2
        term3 = self._b * x3 ** 4 * sin(x1)

        # Return the sum of the terms.
        return term1 + term2 + term3
