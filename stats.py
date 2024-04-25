import scipy.stats as stats
import numpy as np


print(1 - stats.binom.pmf(15, 20, 0.7))
print(stats.binom.cdf(6,10,0.5))
