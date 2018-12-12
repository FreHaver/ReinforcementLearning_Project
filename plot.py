import numpy as np
import matplotlib.pyplot as plt

training_steps = ["500", "1000", "5000", "10000"]
# training_steps = [500, 1000, 5000, 10000]

tab_means = [48.128, 51.212, 108.556, 111.51199999999999]
tab_vars = [184.080496, 245.727416, 923.4632639999998, 876.0429359999998]

deep_means = [87.414, 159.098, 178.334, 176.47400000000002]
deep_vars = [686.9709440000001, 231.05125599999997, 13.88946399999998, 15.740544000000034]

fig = plt.figure(1, figsize=(7, 7))
ax = fig.add_subplot(111)
ax.errorbar(training_steps, tab_means, np.array(tab_vars)**0.5, linestyle='None', marker='o', label="Tabular Dyna-Q", elinewidth=2)
ax.errorbar(training_steps, deep_means, np.array(deep_vars)**0.5, linestyle='None', marker='s', label="Deep Dyna-Q", elinewidth=2)

# ax.xaxis.set_ticks(np.arange(4))
# ax.xaxis.set_ticklabels(training_steps)
ax.legend()
plt.ylim(bottom=0)
plt.ylim(top=200)

plt.show()