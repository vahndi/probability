import matplotlib.pyplot as plt

from probability.models.decision_tree import DecisionTree
from probability.plotting import new_axes

dt = DecisionTree()
# level 1
dt.add_decision(name='D1')
dt.add_option(name='D1.a', p_success=0.7, amount=50, parent_name='D1')
dt.add_option(name='D1.b', p_success=0.5, amount=20, parent_name='D1')
# level 2
dt.add_decision(name='D1.a.D2', parent_name='D1.a')
dt.add_option(name='D1.a.D2.a', p_success=0.8, amount=60,
              parent_name='D1.a.D2', final=True)
dt.add_option(name='D1.a.D2.b', p_success=0.6, amount=25,
              parent_name='D1.a.D2', final=True)
dt.add_decision(name='D1.b.D2', parent_name='D1.b')
dt.add_option(name='D1.b.D2.a', p_success=0.8, amount=60,
              parent_name='D1.b.D2', final=True)
dt.add_option(name='D1.b.D2.b', p_success=0.6, amount=25,
              parent_name='D1.b.D2', final=True)
dt.solve()
ax_names = new_axes()
dt.draw(node_labels='name', ax=ax_names)
plt.show()
ax_amounts = new_axes()
dt.draw(node_labels='amount', ax=ax_amounts)
plt.show()

amounts = dt.amounts()
