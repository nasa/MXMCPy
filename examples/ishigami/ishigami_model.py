from math import sin

import numpy as np


class IshigamiModel:
    """
    Defines Ishigami model with three free params (desc...). The
    quantity of interest that is returned by the evaluate() function is the
    ........................
    """

    def __init__(self, a=7., b=.1, cost=None):

        self._a = a
        self._b = b
        self._cost = cost

    def simulate(self, inputs):
        """
        Run Ishigami function for inputs z1...z3 drawn from
        uniform distributions in the range -pi to pi.
        """
        assert inputs.size == 3
        z1, z2, z3 = inputs

        return self._ishigami_func(z1, z2, z3)

    def evaluate(self, inputs):
        """
        Run Ishigami for each row of inputs and return outputs.
        Each row should have three inputs as defined by Ishigami.
        """
        assert len(inputs.shape) == 2 and inputs.shape[1] == 3

        return np.apply_along_axis(self.simulate, 1, inputs)

    def _ishigami_func(self, z1, z2, z3):
        """
        Return result of Ishigami function for given zi, i=1...3
        """

        # Compute each term.
        term1 = sin(z1)
        term2 = self._a * sin(z2) ** 2
        term3 = self._b * z3 ** 4 * sin(z1)

        # Return the sum of the terms.
        return term1 + term2 + term3
