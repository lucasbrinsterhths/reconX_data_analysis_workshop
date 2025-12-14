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
    'group': np.repeat(['A', 'B'], 10)
})

# manova
ma = MANOVA.from_formula('condition_A + condition_B ~ group', data = df)
print(ma.mv_test())

# Box and whisker plot
plt.boxplot([df.condition_A, df.condition_B], labels=['Condition A', 'Condition B'])
plt.ylabel("Values")
plt.show()
