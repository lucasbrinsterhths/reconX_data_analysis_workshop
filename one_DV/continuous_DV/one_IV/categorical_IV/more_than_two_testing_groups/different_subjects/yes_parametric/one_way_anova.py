# Imports (Using the same for all files in the repo for consistency)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA

# Gernerate sample data for 3 conditions
df = pd.DataFrame({
    'condition_A': np.random.normal(50, 5, 20),
    'condition_B': np.random.normal(60, 5, 20),
    'condition_C': np.random.normal(70, 5, 20)
})

# one-way anova
f, p = stats.f_oneway(df.condition_A, df.condition_B, df.condition_C)
print(f"One-way ANOVA: F={f}, p={p}")

# Box and whisker plot
plt.boxplot([df.condition_A, df.condition_B, df.condition_C], labels=['Condition A', 'Condition B', 'Condition C'])
plt.ylabel("Values")
plt.show()
