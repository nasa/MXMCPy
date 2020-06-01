from math import sin

import numpy as np


class IshigamiModel:
    """
    Defines Ishigami model per paper:
    Title: "Multifidelity Monte Carlo Estimation of Variance and
        Sensitivity Indices"
    Authors: E. Qian, B. Peherstorfer, D. O’Malley, V. V. Vesselinov,
        and K. Willcox
    URL: https://epubs.siam.org/doi/abs/10.1137/17M1151006
    """
    def __init__(self, a=7., b=.1):

        self._a = a
        self._b = b

    def evaluate(self, inputs):
        """
        Run Ishigami for each row of ndarray inputs and return outputs.
        Outputs will be one value for each row in an ndarray.
        Each row should have three inputs as defined by Ishigami.
        """
        assert isinstance(inputs, np.ndarray)
        assert len(inputs.shape) == 2 and inputs.shape[1] == 3

        return np.apply_along_axis(func1d=self.simulate,
                                   axis=1,
                                   arr=inputs)

    def simulate(self, inputs):
        """
        Run Ishigami function for inputs z1...z3 drawn from
        uniform distributions in the range -pi to pi.
        """
        z1, z2, z3 = inputs

        return self._ishigami_func(z1, z2, z3)

    def _ishigami_func(self, z1, z2, z3):
        """
        Return result of Ishigami function for given zi, i=1, 2, 3
        """
        # Compute each term.
        term1 = sin(z1)
        term2 = self._a * sin(z2) ** 2
        term3 = self._b * z3 ** 4 * sin(z1)

        # Return the sum of the terms.
        return term1 + term2 + term3
