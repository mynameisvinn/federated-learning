# federated learning
a tensorflow implementation of ["federated learning: strategies for improving communication efficiency"](https://ai.google/research/pubs/pub45648).

the goal is to learn over distributed devices (eg smartphones), where each device holds data that may be (a) non iid, (b) imbalanced, and (c) sparse.

## stochastic variance reduced gradient (svrg)
[svrg](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf) is core to federated learning. when compared to vanilla sgd, it allows for faster convergence by reducing variance, introduced through small, noisy minibatches.