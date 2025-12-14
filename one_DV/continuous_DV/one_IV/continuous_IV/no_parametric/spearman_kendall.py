# Imports (Using the same for all files in the repo for consistency)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA

# Gernerate sample data for x and y
df = pd.DataFrame({
    'x': np.arange(1, 51)
})

df['y'] = np.log(df.x) + np.random.normal(0, 0.2, 50)

# Spearman correlation
rho, p = stats.spearmanr(df.x, df.y)
print(f"Spearman correlation coefficient: {rho}, p-value: {p}")

# Scatter plot
plt.scatter(df.x, df.y)
plt.xlabel("X"); plt.ylabel("Y")
plt.show()