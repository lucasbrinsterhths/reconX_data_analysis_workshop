# Imports (Using the same for all files in the repo for consistency)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA

# Gernerate sample data for 2 conditions (note the inequal variances)
df = pd.DataFrame({
    'condition_A': np.random.normal(50, 10, 30),
    'condition_B': np.random.normal(60, 30, 30)
})

# wilcoxon matched-pairs signed-rank test
stat, p = stats.wilcoxon(df.condition_A, df.condition_B)
print(f"Wilcoxon signed-rank test: statistic={stat}, p={p}")

# Box and whisker plot
plt.boxplot([df.condition_A, df.condition_B], labels=['Condition A', 'Condition B'])
plt.ylabel("Values")
plt.show()
