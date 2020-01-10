import matplotlib.pyplot as plt
from probability.distributions.multivariate import Dirichlet


Dirichlet(alpha=[2, 2, 2]).pdf().plot_simplex()
plt.show()
Dirichlet(alpha=[20, 2, 2]).pdf().plot_simplex()
plt.show()
