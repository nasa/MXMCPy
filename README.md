# MXMC
main: [![Build Status](https://travis-ci.com/nasa/MXMCPy.svg?branch=main)](https://travis-ci.com/nasa/MXMCPy) [![Coverage Status](https://coveralls.io/repos/github/nasa/MXMCPy/badge.svg?branch=main)](https://coveralls.io/github/nasa/MXMCPy?branch=main)

develop: [![Build Status](https://travis-ci.com/nasa/MXMCPy.svg?branch=develop)](https://travis-ci.com/nasa/MXMCPy) [![Coverage Status](https://coveralls.io/repos/github/nasa/MXMCPy/badge.svg?branch=develop)](https://coveralls.io/github/nasa/MXMCPy?branch=develop) 

## General
MXMCPy is an open source package that implements many existing multi-model 
Monte Carlo methods (MLMC, MFMC, ACV) for estimating statistics from expensive,
high-fidelity models by leveraging faster, low-fidelity models for speedup.

## Getting Started

### Installation

MXMCPy can be easily installed using pip:
```shell
pip install mxmcpy
```
or conda:
```shell
conda install mxmcpy
```

Alternatively, the MXMCPy repository can be cloned:
```shell
git clone https://github.com/nasa/mxmcpy.git
```
and the dependencies can be installed manually as follows. 

### Dependencies
MXMCPy is intended for use with Python 3.x.  MXMCPy requires installation of a 
few dependencies which are relatively common for optimization/numerical methods
with Python:
  - numpy
  - scipy
  - pandas
  - matplotlib
  - h5py
  - pytorch
  - pytest, pytest-mock (if the testing suite is to be run)
  
A `requirements.txt` file is included for easy installation of dependencies with 
`pip` or `conda`.

Installation with pip:
```shell
pip install -r requirements.txt
```

Installation with conda:
```shell
conda install --yes --file requirements.txt
```

### Documentation
Sphynx is used for automatically generating API documentation for MXMCPy. The 
most recent build of the documentation can be found in the repository at: 
`doc/index.html`

## Running Tests
An extensive unit test suite is included with MXMCPy to help ensure proper 
installation. The tests can be run using pytest on the tests directory, e.g., 
by running:
```shell
python -m pytest tests 
```
from the root directory of the repository.

## Example Usage

The following code snippet shows the determination of an optimal sample
allocation for three models with assumed costs and covariance matrix using
the MFMC algorithm:

```python
import numpy as np
from mxmc import Optimizer

model_costs = np.array([1.0, 0.05, 0.001])
covariance_matrix = np.array([[11.531, 11.523, 12.304],
                              [11.523, 11.518, 12.350],
                              [12.304, 12.350, 14.333]])
                             
optimizer = Optimizer(model_costs, covariance_matrix)
opt_result = optimizer.optimize(algorithm="mfmc", target_cost=1000)

print("Optimal variance: ", opt_result.variance)
print("# samples per model: ", opt_result.allocation.get_number_of_samples_per_model())
```

For more detailed examples using MXMCPy including end-to-end construction of
estimators, see the scripts in the [examples directory](examples/). 

## Contributing
1. Fork it (<https://github.com/nasa/mxmcpy/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## Versioning


## Authors
  * Geoffrey Bomarito
  * James Warner
  * Patrick Leser
  * William Leser
  * Luke Morrill
  
## License 

Notices:
Copyright 2020 United States Government as represented by the Administrator of 
the National Aeronautics and Space Administration. No copyright is claimed in 
the United States under Title 17, U.S. Code. All Other Rights Reserved.
 
Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF 
ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED 
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY 
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR 
FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR 
FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE 
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN 
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, 
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS 
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY 
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, 
IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS." 
 
Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE 
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY 
PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY 
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, 
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S 
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE 
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY 
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR 
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS 
AGREEMENT.


