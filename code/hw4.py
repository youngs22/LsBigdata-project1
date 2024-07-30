import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
import matplotlib.pyplot as plt
import seaborn as sns

#n
np.random.seed(20240729)
x = uniform.rvs(loc=3, scale=4, size=10000*20).reshape(-1,20)
k_2 = x.var(axis=1, ddof=0)
population_variance = uniform.var(loc=3, scale=4)

import matplotlib.pyplot as plt
import seaborn as sns
plt.clf()
sns.histplot(k_2, stat='density')
plt.axvline(x=population_variance, color='red')
plt.show()

#n-1
np.random.seed(20240729)
x = uniform.rvs(loc=3, scale=4, size=10000*20).reshape(-1,20)
s_2 = x.var(axis=1, ddof=1)
population_variance = uniform.var(loc=3, scale=4)

plt.clf()
sns.histplot(s_2, stat='density')
plt.axvline(x=population_variance, color='red')
plt.show()
