import numpy as np
import sys
from scipy.integrate import odeint


class SpringMassModel():
    """
    Defines Spring Mass model with 1 free param (stiffness of spring, k). The
    quantity of interest that is returned by the evaluate() function is the
    maximum displacement over the specified time interval
    """
    def __init__(self, mass=1.5, gravity=9.8, state0=None, time_step=None,
                 cost=None):

        self._mass = mass
        self._gravity = gravity

        # Give default initial conditions & time grid if not specified
        if state0 is None:
            state0 = [0.0, 0.0]
        if time_step is None:
            time_grid = np.arange(0.0, 10.0, 0.1)
        else:
            time_grid = np.arange(0.0, 10.0, time_step)

        self._state0 = state0
        self._t = time_grid
        self.cost = cost

    def simulate(self, stiffness):
        """
        Simulate spring mass system for given spring constant. Returns state
        (position, velocity) at all points in time grid
        """
        return odeint(self._integration_func, self._state0, self._t,
                      args=(stiffness, self._mass, self._gravity))

    def evaluate(self, inputs):
        """
        Returns the max displacement over the course of the simulation.
        MXMC convention is that evaluated takes in an array and returns an
        array (even for 1D examples like this one).
        """
        stiffness = inputs[0]
        state = self.simulate(stiffness)
        return np.array([max(state[:, 0])])

    @staticmethod
    def _integration_func(state, t, k, m, g):
        """
        Return velocity/acceleration given velocity/position and values for
        stiffness and mass. Helper function for numerical integrator
        """

        # Unpack the state vector.
        x = state[0]
        xd = state[1]

        # Compute acceleration xdd.
        xdd = ((-k * x) / m) + g

        # Return the two state derivatives.
        return [xd, xdd]


if __name__ == "__main__":

    # Read command line arguments.
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]

    # Parse parameters from input file (convention: gravity, mass,
    # time_step, cost, stiffness).
    inputs = np.genfromtxt(inputfile, delimiter=",", filling_values=None)

    gravity = inputs[0]
    mass = inputs[1]
    time_step = inputs[2]
    cost = inputs[3]
    stiffness = inputs[4]

    # Initialize / evaluate model.
    model = SpringMassModel(mass=mass, gravity=gravity, time_step=time_step)
    max_disp = model.evaluate([stiffness])

    # Write max displacement to output file.
    np.savetxt(outputfile, max_disp)
