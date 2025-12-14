# Imports (Using the same for all files in the repo for consistency)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA

# Gernerate sample data for 3 conditions (note the inequal variances)
df = pd.DataFrame({
    'condition_A': np.random.normal(60, 10, 30),
    'condition_B': np.random.normal(70, 30, 30),
    'condition_C': np.random.normal(80, 20, 30)
})

# friedman anova
stat, p = stats.friedmanchisquare(df.condition_A, df.condition_B, df.condition_C)
print(f"Friedman ANOVA: statistic={stat}, p={p}")

# Box and whisker plot
plt.boxplot([df.condition_A, df.condition_B, df.condition_C], labels=['Condition A', 'Condition B', 'Condition C'])
plt.ylabel("Values")
plt.show()
