# summer-project-2018
This is contains some of my work for the summer project 2018 under supervision by Luigi del Debbio.

## Reports
Unfortunately there were no good reason to finish the reports, thus many of them are still in an unfinished state.
- *01-maxent*: A "paper-like" attempt resulting from a deep-dive into maximum entropy.
- *02-rbms*: A "paper-like" attempt resulting from a deep-dive into RBMs.
- **03-rbm-cumulants-numerical**: This is probably the most (or only) noteworthy of the reports. Here I derive an upper-bound for the terms in the Taylor expansion of the RBM cumulants. This helped to show that extracting the learned couplings of the RBM numerically was not possible, further strengthening the necessity of the analytical expression obtained in https://arxiv.org/abs/1810.11503.

## Notebooks
- *01-MCMC*: From the very beginning of the project. Simple implementations of Metropolis-Hastings and Gibbs sampling.
- *02-RBMs*: First attempt at implementing RBMs. Includes simple implementation of RBMs and application to Ising model.
- **03-toyproblems**: Replicated experiments some experiments the research group focused on using my own RBM package https://github.com/torfjelde/ml. I was also trying to explore the some peculiarities seen when attempting to learn a *discrete* Ising model using a *continuous (Gaussian)* RBM. This was of interest due to the final application (which I cannot discuss here as the group is still working on this).
- **04-upper-bounding-kappa**: Presents failure of the numerical approximation to the Taylor expansion of the RBM cumulants. Work related to *03-rbm-cumulatns-numerical* report. 
