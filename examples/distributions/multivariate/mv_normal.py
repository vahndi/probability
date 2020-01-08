from probability.distributions.multivariate.mv_normal import MVNormal

mvn = MVNormal(mu=[0.5, -0.2], sigma=[[2.0, 0.3], [0.3, 0.5]])
print(mvn.pdf().at([[1, 2], [3, 4]]))
