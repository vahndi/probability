import matplotlib.pyplot as plt
from probability.distributions.multivariate.multinomial import Multinomial
from scipy.stats import multinomial

# scipy
mnsp = multinomial(n=10, p=[0.3, 0.7])
print(mnsp.pmf([3, 7]))  # must sum to 10 (n)
print(mnsp.pmf([4, 6]))  # must sum to 10 (n)
print(mnsp.pmf([5, 5]))  # must sum to 10 (n)
print(mnsp.pmf([[3, 7], [4, 6], [5, 5]]))

# probability
mn = Multinomial(n=10, p=[0.3, 0.7])
print(mn.pmf().at([3, 7]))
print(mn.pmf().at([[3, 7], [4, 6], [5, 5]]))

# visualizations
mn2d = Multinomial(n=10, p=[0.3, 0.7])
mn2d.pmf().plot(mn2d.permutations())
plt.show()
mn3d = Multinomial(n=10, p=[0.3, 0.5, 0.2])
mn3d.pmf().plot(mn3d.permutations())
plt.show()
mn4d = Multinomial(n=10, p=[0.1, 0.2, 0.3, 0.4])
mn4d.pmf().plot(mn4d.permutations())
plt.show()
