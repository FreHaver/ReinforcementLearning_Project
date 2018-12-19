import numpy as np
import matplotlib.pyplot as plt

training_steps = ["500", "1000", "5000", "10000"]
updates = ["2500", "5000", "25000", "50000"]
planning_steps = ["1", "3", "5", "10"]
# training_steps = [500, 1000, 5000, 10000]

# Dyna-Q returns - different training steps / updates
tab_means = [36.44199999999999, 46.672000000000004, 112.83599999999998, 134.07999999999998]
tab_vars = [6.260911754688769, 10.251053409284335, 32.35351764491768, 21.541752017883784]

deep_means = [102.606, 128.596, 175.882, 172.596]
deep_vars = [31.622083802304996, 27.005369540148862, 7.944128397753894, 6.414064545980188]

# Dyna-Q returns - different planning steps (1000 its)
# tab_means_n = [42.635999999999996, 52.172000000000004, 49.516000000000005, 58.138]
# tab_vars_n = [16.404679332434387, 17.74533786660598, 22.09418077232102, 18.350553561132703]
#
# deep_means_n = [109.454, 135.78400000000002, 130.46800000000002, 134.098]
# deep_vars_n = [11.78398506448477, 25.568543642530756, 26.457596565069927, 11.9285244686843]

# 5000 its
tab_means_n = [66.306, 93.106, 103.982, 94.048]
tab_vars_n = [18.612654405000917, 19.24063678779889, 23.722931859279115, 20.742738874121713]

deep_means_n = [153.022, 176.798, 175.882, 153.506]
deep_vars_n = [14.463128845446965, 9.537084250440488, 7.944128397753894, 14.892167874423118]

# interactions
base_means_ints = [32.168, 75.64399999999999, 169.812, 153.876]
base_vars_ints = [0.0, 21.737999999999996, 57.44915194809717, 56.39740947064856]

# updates
base_means_updts = [161.12599999999998, 194.02, 147.922, 152.56]
base_vars_updts = [0.0, 16.447000000000017, 19.383230025520064, 18.01944341537774]


# fig = plt.figure(1, figsize=(7, 7))
# ax = fig.add_subplot(111)
# ax.errorbar(training_steps, tab_means, np.array(tab_vars)**0.5, linestyle='None', marker='o', label="Tabular Dyna-Q", elinewidth=2)
# ax.errorbar(training_steps, deep_means, np.array(deep_vars)**0.5, linestyle='None', marker='s', label="Deep Dyna-Q", elinewidth=2)

plt.errorbar(planning_steps, tab_means_n, np.array(tab_vars_n) / 2, linestyle='None', marker='o', label="Tabular Dyna-Q", elinewidth=2, capsize=5, capthick=2)
plt.errorbar(planning_steps, deep_means_n, np.array(deep_vars_n) / 2, linestyle='None', marker='s', label="Deep Dyna-Q", elinewidth=2, capsize=5, capthick=2)
# plt.errorbar(training_steps, base_means_ints, np.array(base_vars_ints) / 2, linestyle='None', marker='D', label="Baseline", elinewidth=2, capsize=5, capthick=2)
plt.title('Average Episode Lengths on CartPole at Test Time')
plt.xlabel("Number of Planning Steps")
plt.ylabel("Average Episode Length")
plt.legend(loc=4)

# ax.xaxis.set_ticks(np.arange(4))
# ax.xaxis.set_ticklabels(training_steps)
# ax.legend()
plt.ylim(bottom=0)
plt.ylim(top=205)

plt.show()