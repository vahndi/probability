import matplotlib.pyplot as plt
from numpy import arange

from probability.distributions.multivariate.mv_normal import MVNormal

mvn = MVNormal(mu=[0.5, -0.2], sigma=[[2.0, 0.3], [0.3, 0.5]])
print(mvn.pdf().at([[1, 2], [3, 4]]))
x1 = arange(-1, 1.01, 0.01)
x2 = arange(-1, 1.01, 0.01)
mvn.plot_2d(x1=x1, x2=x2)
plt.show()
mvn.cdf().plot_2d(x1=x1, x2=x2, color_map='jet')
plt.show()
