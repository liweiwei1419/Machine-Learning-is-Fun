from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rc('font', family='STHeiti')

samples = stats.chi2.rvs(size=10000, df=1)
plt.plot([1,1],[0,1.5])
sns.distplot(samples)
plt.title('$\chi^2$,df=1')
plt.show()