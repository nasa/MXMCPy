
Introduction
=============

Multi Model Monte Carlo with Python (``MXMCPy``) is a software package developed as a general capability for computing the statistics of outputs from an expensive, high-fidelity model by leveraging faster, low-fidelity models for speedup. Several existing methods are currently implemented, including multi-level Monte Carlo (MLMC) [1], multi-fidelity Monte Carlo (MFMC) [2], and approximate control variates (ACV) [3, 4].  Given a fixed computational budget and a collection of models with varying cost/accuracy,  ``MXMCPy`` will determine an sample allocation strategy across the models that results in an estimator with optimal variance reduction using any of the available algorithms. 


With ``MXMCPy``, users can easily compare existing methods to determine the best choice for their particular problem, while developers have a basis for implementing and sharing new variance reduction approaches. See the remainder of the documentation for more details of using the code. For additional information, see the report that accompanied the release of ``MXMCPy`` [5].


| [1] Giles, M. B.: Multi-level Monte Carlo path simulation. OPERATIONS RESEARCH , vol. 56, no. 3, 2008, pp. 607–617.
| [2] Peherstorfer, B.; Willcox, K.; and Gunzburger, M.: Optimal Model Management for Multifidelity Monte Carlo Estimation. SIAM Journal on Scientific Computing, vol. 38, 01 2016, pp. A3163–A3194
| [3] Gorodetsky, A.; Geraci, G.; Eldred, M.; and Jakeman, J. D.: A generalized approximate control variate framework for multifidelity uncertainty quantification. Journal of Computational Physics, 2020, p. 109257
| [4] Bomarito, G. F., Leser, P. E., Warner, J. E., and Leser, W. P: On the Optimization of Approximate Control Variates with Parametrically-Defined Estimators. In Preparation.
| [5] Bomarito, G. F., Warner, J. E., Leser, P. E., Leser, W. P., and Morrill, L.: Multi Model Monte Carlo with Python (MXMCPy). NASA/TM–2020–220585. 2020.
