Available Algorithms
====================

There are currently 13 optimization algorithms implemented in ``MXMCPy``.  The algorithms are listed below along with some general information.

Multi-level Monte Carlo
-----------------------

+------------------+--------------+----------------------+
| Algorithm Name   | Optimization | Sampling Strategy    |
+==================+==============+======================+
| MLMC             | Analytic     | Recursive Difference |
+------------------+--------------+----------------------+

Ref: Giles, M. B.: Multi-level Monte Carlo path simulation. Operations Research , vol. 56, no. 3, 2008, pp. 607–617.


Multifidelity Monte Carlo
-------------------------

+------------------+--------------+----------------------+
| Algorithm Name   | Optimization | Sampling Strategy    |
+==================+==============+======================+
| MFMC             | Analytic     | Multifidelity        |
+------------------+--------------+----------------------+

Ref: Peherstorfer, B.; Willcox, K.; and Gunzburger, M.: Optimal Model Management for Multifidelity Monte Carlo Estimation. SIAM Journal on Scientific Computing, vol. 38, 01 2016, pp. A3163–A3194


Approximate Control Variates
----------------------------

+------------------+--------------+----------------------+
| Algorithm Name   | Optimization | Sampling Strategy    |
+==================+==============+======================+
| WRDIFF           | Numerical    | Recursive Difference |
+------------------+--------------+----------------------+
| ACVIS            | Numerical    | Independent Samples  |
+------------------+--------------+----------------------+
| ACVMF            | Numerical    | Multifidelity        |
+------------------+--------------+----------------------+
| ACVKL            | Numerical    | Multifidelity        |
+------------------+--------------+----------------------+

Ref: Gorodetsky, A.; Geraci, G.; Eldred, M.; and Jakeman, J. D.: A generalized approximate control variate framework for multifidelity uncertainty quantification. Journal of Computational Physics, 2020, p. 109257


Parametrically-defined Approximate Control Variates
---------------------------------------------------

+------------------+--------------+----------------------+
| Algorithm Name   | Optimization | Sampling Strategy    |
+==================+==============+======================+
| GRDSR            | Numerical    | Recursive Difference |
+------------------+--------------+----------------------+
| GRDMR            | Numerical    | Recursive Difference |
+------------------+--------------+----------------------+
| GISSR            | Numerical    | Independent Samples  |
+------------------+--------------+----------------------+
| GISMR            | Numerical    | Independent Samples  |
+------------------+--------------+----------------------+
| ACVMFU           | Numerical    | Multifidelity        |
+------------------+--------------+----------------------+
| GMFSR            | Numerical    | Multifidelity        |
+------------------+--------------+----------------------+
| GMFMR            | Numerical    | Multifidelity        |
+------------------+--------------+----------------------+

Ref: Bomarito, G. F., Leser, P. E., Warner, J. E., and Leser, W. P: On the Optimization of Approximate Control Variates with Parametrically-Defined Estimators. In Preparation.
